#!/usr/bin/env python
"""
ETA-based alerting with LLM explanation.

- Given:
  - vehicle current location (lat, lon)
  - destination (lat, lon)
  - estimated arrival datetime (ETA, in ISO format)
  - assumed average speed (km/h)

- Logic:
  1. Compute great-circle distance (Haversine).
  2. Compute travel_time_needed = distance / avg_speed.
  3. Compare with remaining_time = ETA - now.
  4. If travel_time_needed > remaining_time (with optional slack), mark alert=True.
  5. Use ChatOpenAI to generate a natural language explanation and recommendation.

Requires:
    pip install langchain langchain-openai python-dotenv
    export OPENAI_API_KEY=...   (or .env file)
"""

import os
import math
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ----------------------------------------------------
# Config
# ----------------------------------------------------

# Average speed (km/h) – you can make this dynamic based on route/history
DEFAULT_AVG_SPEED_KMPH = 60.0

# Safety factor: we can require that we have, say, 10% extra time than pure distance/speed
SAFETY_TIME_FACTOR = 1.10  # 10% buffer

# Load env variables (OPENAI_API_KEY)
load_dotenv()


# ----------------------------------------------------
# Data Models
# ----------------------------------------------------

@dataclass
class VehicleETAInput:
    vehicle_id: str
    current_lat: float
    current_lon: float
    dest_lat: float
    dest_lon: float
    eta_iso: str              # Estimated arrival datetime in ISO 8601 format, e.g. "2025-12-06T18:00:00"
    avg_speed_kmph: float = DEFAULT_AVG_SPEED_KMPH


@dataclass
class ETAEvaluationResult:
    vehicle_id: str
    distance_km: float
    travel_time_needed_hours: float
    remaining_time_hours: float
    will_arrive_on_time: bool
    alert: bool
    reason: str
    llm_explanation: Optional[str] = None


# ----------------------------------------------------
# Utility Functions
# ----------------------------------------------------

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on the Earth (in km).
    """
    R = 6371.0  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def evaluate_eta(input_data: VehicleETAInput, now: Optional[datetime] = None) -> ETAEvaluationResult:
    """
    Pure deterministic logic to decide if the vehicle is on track.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        eta_dt = datetime.fromisoformat(input_data.eta_iso)
        if eta_dt.tzinfo is None:
            # assume UTC if no timezone is given
            eta_dt = eta_dt.replace(tzinfo=timezone.utc)
    except Exception as e:
        raise ValueError(f"Invalid eta_iso format: {input_data.eta_iso} ({e})")

    distance_km = haversine_distance_km(
        input_data.current_lat,
        input_data.current_lon,
        input_data.dest_lat,
        input_data.dest_lon,
    )

    # time needed in hours at avg speed
    if input_data.avg_speed_kmph <= 0:
        raise ValueError("avg_speed_kmph must be positive")
    travel_time_needed_hours = distance_km / input_data.avg_speed_kmph

    # remaining time until ETA (can be negative if already past ETA)
    delta = eta_dt - now
    remaining_time_hours = delta.total_seconds() / 3600.0

    # incorporate safety factor
    travel_time_with_safety = travel_time_needed_hours * SAFETY_TIME_FACTOR

    # decision
    if remaining_time_hours <= 0:
        will_arrive_on_time = False
        alert = True
        reason = "ETA is in the past or exactly now; vehicle has not yet arrived."
    else:
        will_arrive_on_time = travel_time_with_safety <= remaining_time_hours
        alert = not will_arrive_on_time
        if alert:
            reason = (
                f"Required travel time with safety buffer ({travel_time_with_safety:.2f} h) "
                f"is greater than remaining time to ETA ({remaining_time_hours:.2f} h)."
            )
        else:
            reason = (
                f"Required travel time with safety buffer ({travel_time_with_safety:.2f} h) "
                f"is less than or equal to remaining time to ETA ({remaining_time_hours:.2f} h)."
            )

    return ETAEvaluationResult(
        vehicle_id=input_data.vehicle_id,
        distance_km=distance_km,
        travel_time_needed_hours=travel_time_needed_hours,
        remaining_time_hours=remaining_time_hours,
        will_arrive_on_time=will_arrive_on_time,
        alert=alert,
        reason=reason,
    )


# ----------------------------------------------------
# LLM Explanation using ChatOpenAI
# ----------------------------------------------------

def build_llm() -> ChatOpenAI:
    """
    Build a ChatOpenAI instance (LangChain).
    Make sure OPENAI_API_KEY is set in env or .env.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    llm = ChatOpenAI(
        model="gpt-4o-mini",   # or "gpt-4o", "gpt-3.5-turbo", etc.
        temperature=0.2,
    )
    return llm


def generate_llm_explanation(llm: ChatOpenAI, eval_result: ETAEvaluationResult) -> str:
    """
    Use the LLM to generate a user-friendly explanation and recommendation based
    on the evaluation result.
    """
    system_msg = (
        "You are a logistics assistant helping monitor delivery trucks.\n"
        "Given current location, destination, estimated arrival time, and distance/time calculations, "
        "you must explain whether the vehicle is on time or at risk of delay. "
        "If there is a risk, suggest proactive actions (e.g., rescheduling, notifying customer, "
        "asking driver to increase speed safely, etc.). "
        "Be concise (5-10 sentences) and non-technical."
    )

    data = asdict(eval_result)
    user_msg = (
        f"Here are the computed details for a truck's ETA:\n"
        f"Vehicle ID: {data['vehicle_id']}\n"
        f"Distance to destination: {data['distance_km']:.2f} km\n"
        f"Estimated travel time needed (at planned average speed): "
        f"{data['travel_time_needed_hours']:.2f} hours\n"
        f"Remaining time until planned ETA: {data['remaining_time_hours']:.2f} hours\n"
        f"Will arrive on time (according to the calculation): {data['will_arrive_on_time']}\n"
        f"Alert flag: {data['alert']}\n"
        f"Reason: {data['reason']}\n\n"
        f"Explain in clear terms whether this truck is on track to reach the destination on time, "
        f"and what actions, if any, the operations team should take now."
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    )
    return resp.content


# ----------------------------------------------------
# Example usage (CLI-style)
# ----------------------------------------------------

def main():
    # Example input: adjust these with real data or wire to REST API / DB
    input_data = VehicleETAInput(
        vehicle_id="TRUCK-123",
        current_lat=19.0760,      # e.g., Mumbai
        current_lon=72.8777,
        dest_lat=18.5204,         # e.g., Pune
        dest_lon=73.8567,
        eta_iso="2025-12-06T20:00:00+05:30",  # estimated arrival time (IST in this example)
        avg_speed_kmph=50.0,
    )

    # 1) Pure math-based evaluation
    eval_result = evaluate_eta(input_data)

    # 2) Get LLM explanation
    llm = build_llm()
    explanation = generate_llm_explanation(llm, eval_result)
    eval_result.llm_explanation = explanation

    # 3) Print or send alert
    print("=== Evaluation Summary (Structured) ===")
    for k, v in asdict(eval_result).items():
        if k != "llm_explanation":
            print(f"{k}: {v}")

    print("\n=== LLM Explanation ===")
    print(explanation)

    if eval_result.alert:
        print("\n*** ALERT: Vehicle is at risk of missing ETA. ***")


if __name__ == "__main__":
    main()
