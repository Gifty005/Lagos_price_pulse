from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RiskAssessment:
    current_price: float
    predicted_price: float
    pct_change: float
    status: str
    headline: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_price_risk(
    current_price: float, predicted_price: float
) -> Dict[str, Any]:
    """
    Evaluates next-month commodity price risk based on projected percentage change.

    Tiers:
      - GREEN:  pct_change < 5.0%
      - YELLOW: 5.0% <= pct_change <= 15.0%
      - RED:    pct_change > 15.0%
    """
    if current_price <= 0:
        raise ValueError(
            f"current_price must be greater than zero, got {current_price}"
        )

    pct_change = ((predicted_price - current_price) / current_price) * 100.0

    if pct_change < 5.0:
        status = "GREEN"
        headline = "Low Risk / Stable Market"
        recommendation = (
            "Normal market conditions; safe to follow regular buying and "
            "storage schedules."
        )
    elif pct_change <= 15.0:
        status = "YELLOW"
        headline = "Moderate Risk / Upward Price Pressure"
        recommendation = (
            "Moderate upward price pressure; stagger wholesale orders and "
            "monitor diesel and FX shifts closely."
        )
    else:
        status = "RED"
        headline = "High Risk / Severe Price Spike Alert"
        recommendation = (
            "Severe price spike expected; bulk buyers and households should "
            "procure early before prices surge."
        )

    assessment = RiskAssessment(
        current_price=round(float(current_price), 2),
        predicted_price=round(float(predicted_price), 2),
        pct_change=round(float(pct_change), 2),
        status=status,
        headline=headline,
        recommendation=recommendation,
    )
    return assessment.to_dict()