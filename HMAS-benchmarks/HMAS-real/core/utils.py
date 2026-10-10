# ─────────────────────────────────────────────────────────
# utils.py
# Role : utility functions shared across agents
#
# Centralized here to avoid duplication between
# planner_executer.py and verifier_agent.py.
# ─────────────────────────────────────────────────────────

import json
import os

TASK_TYPE_MAP = {
    "dashdish":    "food_ordering",
    "fly-unified": "travel_booking",
    "gocalendar":  "calendar_management",
    "gomail":      "email_management",
    "networkin":   "social_networking",
    "omnizon":     "online_shopping",
    "opendining":  "restaurant_booking",
    "staynb":      "accommodation_booking",
    "topwork":     "job_platform",
    "udriver":     "ride_sharing",
    "zilloft":     "real_estate",
    "marrisuite":  "hotel_management",
}


def extract_task_type(task_id: str) -> str:
    """
    Extracts the business type of a task from its ID.

    Allows grouping learned strategies by domain :
    all dashdish tasks share the same learned food_ordering
    strategies, even if the tasks themselves differ.

    Examples :
      "v1.dashdish-6"  -> "food_ordering"
      "v2.networkin-3" -> "social_networking"
      "v1.unknown-1"   -> "general_web"
    """
    clean    = task_id.replace("v1.", "").replace("v2.", "")
    category = clean.split("-")[0]
    return TASK_TYPE_MAP.get(category, "general_web")
