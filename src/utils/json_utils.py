import json
import re
from typing import Any, Dict, Optional


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from model output.
    Works with raw JSON or JSON wrapped in markdown fences.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


def normalize_response_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make sure the parsed response has a predictable shape.
    """
    response_type = obj.get("type", "final")

    if response_type == "tool_call":
        return {
            "type": "tool_call",
            "tool_name": obj.get("tool_name", ""),
            "arguments": obj.get("arguments", {}) or {},
        }

    return {
        "type": "final",
        "answer": obj.get("answer", "") or obj.get("content", "") or "",
    }