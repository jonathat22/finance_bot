"""Value investing models for intrinsic value analysis."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class IntrinsicValueEstimate(BaseModel):
    """Result of a single valuation method."""

    ticker: str
    method: Literal["dcf", "graham_number", "owner_earnings"]
    intrinsic_value_per_share: float
    current_price: float
    margin_of_safety_pct: float = Field(
        description="(intrinsic - current) / intrinsic as percentage"
    )
    assumptions: Dict[str, Any] = Field(
        default_factory=dict, description="Method-specific parameters"
    )


class MoatAssessment(BaseModel):
    """Economic moat evaluation."""

    ticker: str
    moat_rating: Literal["wide", "narrow", "none"]
    moat_sources: List[str] = Field(
        default_factory=list,
        description="e.g., ['brand', 'switching_costs', 'network_effects']",
    )
    gross_margin_stability: float = Field(
        description="Std dev of gross profit margin over available years"
    )
    avg_competitive_advantage_gap: Optional[float] = None
    summary: str = ""


class EarningsConsistency(BaseModel):
    """Earnings predictability analysis."""

    ticker: str
    eps_history: List[Optional[float]]
    years: List[int]
    earnings_growth_cagr: Optional[float] = None
    positive_earnings_streak: int = 0
    coefficient_of_variation: Optional[float] = None
    is_predictable: bool = False


class DebtAnalysis(BaseModel):
    """Balance sheet safety assessment."""

    ticker: str
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    interest_coverage: Optional[float] = None
    debt_to_earnings: Optional[float] = None
    rating: Literal["conservative", "moderate", "aggressive", "dangerous"] = "moderate"


class ValueScorecard(BaseModel):
    """Complete value investing assessment for a company."""

    ticker: str
    company_name: str
    current_price: float
    intrinsic_values: List[IntrinsicValueEstimate] = Field(default_factory=list)
    moat: Optional[MoatAssessment] = None
    earnings: Optional[EarningsConsistency] = None
    debt: Optional[DebtAnalysis] = None
    overall_rating: Literal["strong_buy", "buy", "hold", "avoid"] = "hold"
    verdict: str = Field(default="", description="Concise investment thesis")
    key_strengths: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)
