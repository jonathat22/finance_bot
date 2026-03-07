"""ValuationAgent: Compute intrinsic value using DCF, Graham Number, Owner Earnings.

Also produces MoatAssessment, EarningsConsistency, DebtAnalysis, and
a composite ValueScorecard with overall rating.
"""

import argparse
import json
import math
import statistics
import sys
from typing import Any, Dict, List, Optional

from finance_bot.models.valuation import (
    DebtAnalysis,
    EarningsConsistency,
    IntrinsicValueEstimate,
    MoatAssessment,
    ValueScorecard,
)


# Conservative defaults per value-investing-framework.md
MAX_GROWTH_RATE = 0.08
DISCOUNT_RATE = 0.10
TERMINAL_GROWTH = 0.03
PROJECTION_YEARS = 10


class ValuationAgent:
    """Computes intrinsic value and produces a ValueScorecard."""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[ValuationAgent] {msg}", file=sys.stderr)

    def compute_scorecard(self, data: Dict[str, Any]) -> ValueScorecard:
        """Build a complete ValueScorecard from DataAgent output."""
        ticker = data["ticker"]
        company_name = data.get("company_name", ticker)
        current_price = data.get("current_price") or 0.0

        intrinsic_values = []
        # DCF
        dcf = self.dcf_valuation(data)
        if dcf:
            intrinsic_values.append(dcf)

        # Graham Number
        graham = self.graham_number(data)
        if graham:
            intrinsic_values.append(graham)

        # Owner Earnings
        oe = self.owner_earnings_valuation(data)
        if oe:
            intrinsic_values.append(oe)

        moat = self.assess_moat(data)
        earnings = self.assess_earnings_consistency(data)
        debt = self.assess_debt(data)

        rating = self._compute_rating(intrinsic_values, moat, earnings, debt)
        verdict = self._generate_verdict(
            ticker, rating, intrinsic_values, moat, current_price
        )
        strengths, risks = self._identify_strengths_risks(
            data, intrinsic_values, moat, earnings, debt
        )

        return ValueScorecard(
            ticker=ticker,
            company_name=company_name,
            current_price=current_price,
            intrinsic_values=intrinsic_values,
            moat=moat,
            earnings=earnings,
            debt=debt,
            overall_rating=rating,
            verdict=verdict,
            key_strengths=strengths,
            key_risks=risks,
        )

    def dcf_valuation(self, data: Dict[str, Any]) -> Optional[IntrinsicValueEstimate]:
        """DCF: project FCF forward, discount back, add terminal value."""
        ticker = data["ticker"]
        current_price = data.get("current_price") or 0.0
        shares = data.get("shares_outstanding")
        fcf_data = data.get("free_cash_flow", {})
        fcf_list = fcf_data.get("fcf", [])

        # Need at least one FCF and shares outstanding
        valid_fcf = [f for f in fcf_list if f is not None and f > 0]
        if not valid_fcf or not shares or shares <= 0:
            self._log("DCF: insufficient data")
            return None

        base_fcf = valid_fcf[-1]  # Most recent

        # Compute historical FCF growth rate
        if len(valid_fcf) >= 2:
            growth = (valid_fcf[-1] / valid_fcf[0]) ** (1 / (len(valid_fcf) - 1)) - 1
            growth = min(growth, MAX_GROWTH_RATE)  # Cap at 8%
            growth = max(growth, 0.0)  # Floor at 0%
        else:
            growth = 0.03  # Default conservative

        # Project and discount
        total_pv = 0.0
        projected_fcf = base_fcf
        for year in range(1, PROJECTION_YEARS + 1):
            projected_fcf *= 1 + growth
            pv = projected_fcf / (1 + DISCOUNT_RATE) ** year
            total_pv += pv

        # Terminal value
        terminal_fcf = projected_fcf * (1 + TERMINAL_GROWTH)
        terminal_value = terminal_fcf / (DISCOUNT_RATE - TERMINAL_GROWTH)
        pv_terminal = terminal_value / (1 + DISCOUNT_RATE) ** PROJECTION_YEARS
        total_pv += pv_terminal

        intrinsic_per_share = total_pv / shares
        mos = (
            (intrinsic_per_share - current_price) / intrinsic_per_share * 100
            if intrinsic_per_share > 0
            else -100.0
        )

        return IntrinsicValueEstimate(
            ticker=ticker,
            method="dcf",
            intrinsic_value_per_share=round(intrinsic_per_share, 2),
            current_price=current_price,
            margin_of_safety_pct=round(mos, 1),
            assumptions={
                "discount_rate": DISCOUNT_RATE,
                "growth_rate": round(growth, 4),
                "terminal_growth": TERMINAL_GROWTH,
                "base_fcf": base_fcf,
                "projection_years": PROJECTION_YEARS,
            },
        )

    def graham_number(self, data: Dict[str, Any]) -> Optional[IntrinsicValueEstimate]:
        """Graham Number: sqrt(22.5 * EPS * BVPS)."""
        ticker = data["ticker"]
        current_price = data.get("current_price") or 0.0

        # Get EPS
        metrics = data.get("metrics", {})
        eps = metrics.get("eps")

        # Get Book Value Per Share
        bvps = data.get("book_value_per_share")

        if not eps or eps <= 0 or not bvps or bvps <= 0:
            self._log("Graham: EPS or BVPS not positive")
            return None

        graham_value = math.sqrt(22.5 * eps * bvps)
        mos = (
            (graham_value - current_price) / graham_value * 100
            if graham_value > 0
            else -100.0
        )

        return IntrinsicValueEstimate(
            ticker=ticker,
            method="graham_number",
            intrinsic_value_per_share=round(graham_value, 2),
            current_price=current_price,
            margin_of_safety_pct=round(mos, 1),
            assumptions={"eps": eps, "bvps": bvps, "multiplier": 22.5},
        )

    def owner_earnings_valuation(
        self, data: Dict[str, Any]
    ) -> Optional[IntrinsicValueEstimate]:
        """Owner Earnings: (NI + D&A - avg CapEx) / discount rate / shares."""
        ticker = data["ticker"]
        current_price = data.get("current_price") or 0.0
        shares = data.get("shares_outstanding")
        oe_data = data.get("owner_earnings", {})
        owner_earnings = oe_data.get("owner_earnings")

        if not owner_earnings or owner_earnings <= 0 or not shares or shares <= 0:
            self._log("Owner Earnings: insufficient data")
            return None

        intrinsic_total = owner_earnings / DISCOUNT_RATE
        intrinsic_per_share = intrinsic_total / shares
        mos = (
            (intrinsic_per_share - current_price) / intrinsic_per_share * 100
            if intrinsic_per_share > 0
            else -100.0
        )

        return IntrinsicValueEstimate(
            ticker=ticker,
            method="owner_earnings",
            intrinsic_value_per_share=round(intrinsic_per_share, 2),
            current_price=current_price,
            margin_of_safety_pct=round(mos, 1),
            assumptions={
                "net_income": oe_data.get("net_income"),
                "depreciation": oe_data.get("depreciation_amortization"),
                "avg_capex_3yr": oe_data.get("avg_capex_3yr"),
                "discount_rate": DISCOUNT_RATE,
            },
        )

    def assess_moat(self, data: Dict[str, Any]) -> Optional[MoatAssessment]:
        """Quantitative moat assessment from margin stability."""
        ticker = data["ticker"]
        margins = data.get("margins")
        if not margins:
            return None

        gpms = [g for g in margins.get("gross_profit_margins", []) if g is not None]
        if len(gpms) < 2:
            return None

        gpm_std = statistics.stdev(gpms)
        avg_gap = margins.get("competitive_advantage_gaps", [])
        avg_gap_vals = [g for g in avg_gap if g is not None]
        avg_gap_val = (
            sum(avg_gap_vals) / len(avg_gap_vals) if avg_gap_vals else None
        )

        # Classify moat
        moat_sources = []
        if gpm_std < 0.03 and avg_gap_val is not None and avg_gap_val > 0.10:
            rating = "wide"
        elif gpm_std < 0.05 and avg_gap_val is not None and avg_gap_val > 0.05:
            rating = "narrow"
        else:
            rating = "none"

        # Infer moat sources from metrics
        metrics = data.get("metrics", {})
        profit_margin = metrics.get("profit_margin")
        if profit_margin and profit_margin > 0.20:
            moat_sources.append("pricing_power")
        if avg_gap_val and avg_gap_val > 0.10:
            moat_sources.append("cost_advantage")
        revenue_growth = metrics.get("revenue_growth")
        if revenue_growth and revenue_growth > 0.10:
            moat_sources.append("growth")

        summary = f"GPM std dev: {gpm_std:.3f}. "
        if avg_gap_val is not None:
            summary += f"Avg competitive advantage gap: {avg_gap_val:.1%}. "
        summary += f"Rating: {rating}."

        return MoatAssessment(
            ticker=ticker,
            moat_rating=rating,
            moat_sources=moat_sources,
            gross_margin_stability=round(gpm_std, 4),
            avg_competitive_advantage_gap=(
                round(avg_gap_val, 4) if avg_gap_val is not None else None
            ),
            summary=summary,
        )

    def assess_earnings_consistency(
        self, data: Dict[str, Any]
    ) -> Optional[EarningsConsistency]:
        """Analyze EPS history for predictability."""
        ticker = data["ticker"]
        eps_data = data.get("eps_history", {})
        years = eps_data.get("years", [])
        eps_list = eps_data.get("eps", [])

        if not years or not eps_list:
            return None

        import math

        valid_eps = [
            (y, e)
            for y, e in zip(years, eps_list)
            if e is not None and not (isinstance(e, float) and math.isnan(e))
        ]
        if not valid_eps:
            return None

        eps_vals = [e for _, e in valid_eps]
        positive_streak = 0
        for e in reversed(eps_vals):
            if e > 0:
                positive_streak += 1
            else:
                break

        # CAGR
        cagr = None
        if len(eps_vals) >= 2 and eps_vals[0] > 0 and eps_vals[-1] > 0:
            n_years = len(eps_vals) - 1
            if n_years > 0:
                cagr = (eps_vals[-1] / eps_vals[0]) ** (1 / n_years) - 1

        # Coefficient of variation
        cv = None
        if len(eps_vals) >= 3:
            mean_eps = statistics.mean(eps_vals)
            if mean_eps != 0:
                cv = statistics.stdev(eps_vals) / abs(mean_eps)

        is_predictable = (cv is not None and cv < 0.50) and positive_streak >= 5

        return EarningsConsistency(
            ticker=ticker,
            eps_history=eps_list,
            years=years,
            earnings_growth_cagr=round(cagr, 4) if cagr is not None else None,
            positive_earnings_streak=positive_streak,
            coefficient_of_variation=round(cv, 4) if cv is not None else None,
            is_predictable=is_predictable,
        )

    def assess_debt(self, data: Dict[str, Any]) -> Optional[DebtAnalysis]:
        """Classify debt safety from balance sheet ratios."""
        ticker = data["ticker"]
        metrics = data.get("metrics", {})

        de = metrics.get("debt_to_equity")
        # yfinance returns D/E as percentage (e.g., 173 = 1.73x)
        if de is not None and de > 10:
            de = de / 100.0

        cr = metrics.get("current_ratio")
        ic = data.get("interest_coverage")

        # Debt to earnings
        eps = metrics.get("eps")
        dte = None
        if de is not None and eps is not None and eps > 0:
            dte = de / eps if eps != 0 else None

        # Rating logic
        rating = "moderate"
        if de is not None:
            if de < 0.5 and (cr is None or cr > 2.0) and (ic is None or ic > 8):
                rating = "conservative"
            elif de > 2.5 or (cr is not None and cr < 1.0) or (ic is not None and ic < 2):
                rating = "dangerous"
            elif de > 1.5 or (cr is not None and cr < 1.5) or (ic is not None and ic < 4):
                rating = "aggressive"

        return DebtAnalysis(
            ticker=ticker,
            debt_to_equity=round(de, 2) if de is not None else None,
            current_ratio=round(cr, 2) if cr is not None else None,
            interest_coverage=round(ic, 2) if ic is not None else None,
            debt_to_earnings=round(dte, 2) if dte is not None else None,
            rating=rating,
        )

    def _compute_rating(
        self,
        intrinsic_values: List[IntrinsicValueEstimate],
        moat: Optional[MoatAssessment],
        earnings: Optional[EarningsConsistency],
        debt: Optional[DebtAnalysis],
    ) -> str:
        """Composite rating from all assessments."""
        if not intrinsic_values:
            return "hold"

        avg_mos = statistics.mean([iv.margin_of_safety_pct for iv in intrinsic_values])

        has_wide_moat = moat and moat.moat_rating == "wide"
        has_narrow_moat = moat and moat.moat_rating == "narrow"
        is_predictable = earnings and earnings.is_predictable
        debt_ok = debt is None or debt.rating in ("conservative", "moderate")
        debt_bad = debt and debt.rating == "dangerous"
        no_moat = moat and moat.moat_rating == "none"

        if avg_mos > 50 and has_wide_moat and is_predictable and debt_ok:
            return "strong_buy"
        elif avg_mos > 30 and (has_wide_moat or has_narrow_moat):
            return "buy"
        elif avg_mos < 0 and (debt_bad or no_moat):
            return "avoid"
        else:
            return "hold"

    def _generate_verdict(
        self,
        ticker: str,
        rating: str,
        intrinsic_values: List[IntrinsicValueEstimate],
        moat: Optional[MoatAssessment],
        current_price: float,
    ) -> str:
        """Generate a concise investment thesis."""
        if not intrinsic_values:
            return f"Insufficient data for {ticker} valuation."

        avg_mos = statistics.mean([iv.margin_of_safety_pct for iv in intrinsic_values])
        moat_desc = moat.moat_rating if moat else "unknown"

        if rating == "strong_buy":
            return (
                f"{ticker} trades at a significant discount to intrinsic value "
                f"({avg_mos:.0f}% margin of safety) with a {moat_desc} moat. "
                f"Strong buy candidate for value investors."
            )
        elif rating == "buy":
            return (
                f"{ticker} offers adequate margin of safety ({avg_mos:.0f}%) "
                f"with a {moat_desc} moat. Attractive at current levels."
            )
        elif rating == "avoid":
            return (
                f"{ticker} appears overvalued ({avg_mos:.0f}% MOS) "
                f"with {moat_desc} moat characteristics. "
                f"Insufficient margin of safety for new positions."
            )
        else:
            return (
                f"{ticker} at ${current_price:.2f} trades near fair value "
                f"({avg_mos:.0f}% MOS). {moat_desc.capitalize()} moat. "
                f"Hold existing positions; wait for better entry."
            )

    def _identify_strengths_risks(
        self,
        data: Dict[str, Any],
        intrinsic_values: List[IntrinsicValueEstimate],
        moat: Optional[MoatAssessment],
        earnings: Optional[EarningsConsistency],
        debt: Optional[DebtAnalysis],
    ) -> tuple:
        """Identify key strengths and risks."""
        strengths = []
        risks = []

        if moat:
            if moat.moat_rating == "wide":
                strengths.append("Wide economic moat")
            elif moat.moat_rating == "none":
                risks.append("No identifiable economic moat")

        if earnings:
            if earnings.is_predictable:
                cagr_str = (
                    f" ({earnings.earnings_growth_cagr:.1%} CAGR)"
                    if earnings.earnings_growth_cagr
                    else ""
                )
                strengths.append(
                    f"Predictable earnings, {earnings.positive_earnings_streak}yr positive streak{cagr_str}"
                )
            else:
                risks.append("Unpredictable or volatile earnings")

        if debt:
            if debt.rating == "conservative":
                strengths.append("Conservative balance sheet")
            elif debt.rating in ("aggressive", "dangerous"):
                risks.append(f"Elevated debt levels ({debt.rating})")

        metrics = data.get("metrics", {})
        profit_margin = metrics.get("profit_margin")
        if profit_margin and profit_margin > 0.20:
            strengths.append(f"Strong profitability ({profit_margin:.1%} net margin)")

        if intrinsic_values:
            avg_mos = statistics.mean(
                [iv.margin_of_safety_pct for iv in intrinsic_values]
            )
            if avg_mos > 30:
                strengths.append(f"Trading below intrinsic value ({avg_mos:.0f}% MOS)")
            elif avg_mos < 0:
                risks.append(
                    f"Trading above intrinsic value ({abs(avg_mos):.0f}% premium)"
                )

        return strengths, risks


def main():
    parser = argparse.ArgumentParser(description="Value investing analysis")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument("--data", required=True, help="JSON data from DataAgent")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.data)
    agent = ValuationAgent(verbose=args.verbose)
    scorecard = agent.compute_scorecard(data)
    print(scorecard.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
