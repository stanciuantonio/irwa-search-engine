import re
from typing import Any, Mapping, Optional


def parse_rating(value: Any) -> Optional[float]:
    """
    Convert a raw rating value (e.g. '3.9', '', None) to float or None.
    Mirrors the logic in myapp.search.objects.Document.parse_rating.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return float(value)


def parse_discount(value: Any) -> Optional[float]:
    """
    Convert a raw discount value (e.g. '69% off', '30', None) to a float
    representing the percentage (e.g. 69.0) or None.
    Mirrors the logic in myapp.search.objects.Document.parse_discount.
    """
    if value is None:
        return None
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value.replace(",", ""))
        if match:
            return float(match.group(1))
        return None
    return float(value)


def rating_norm(original_doc: Mapping[str, Any]) -> float:
    """
    Normalized rating in [0,1] from original_doc['average_rating'].
    Returns 0.0 when rating is missing or invalid.
    """
    raw = original_doc.get("average_rating")
    rating = parse_rating(raw)
    if rating is None:
        return 0.0
    # typical range 0–5
    return max(0.0, min(rating / 5.0, 1.0))


def discount_norm(original_doc: Mapping[str, Any]) -> float:
    """
    Normalized discount in [0,1] from original_doc['discount'].
    Returns 0.0 when discount is missing or invalid.
    """
    raw = original_doc.get("discount")
    discount = parse_discount(raw)
    if discount is None:
        return 0.0
    # assume discount is a percentage 0–100
    return max(0.0, min(discount / 100.0, 1.0))
