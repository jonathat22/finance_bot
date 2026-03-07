"""Tests for value investing models."""

import pytest

from finance_bot.models.valuation import (
    DebtAnalysis,
    EarningsConsistency,
    IntrinsicValueEstimate,
    MoatAssessment,
    ValueScorecard,
)


class TestIntrinsicValueEstimate:
    def test_positive_margin_of_safety(self):
        estimate = IntrinsicValueEstimate(
            ticker="AAPL",
            method="dcf",
            intrinsic_value_per_share=200.0,
            current_price=150.0,
            margin_of_safety_pct=25.0,
            assumptions={"discount_rate": 0.10, "growth_rate": 0.065},
        )
        assert estimate.margin_of_safety_pct == 25.0
        assert estimate.method == "dcf"

    def test_negative_margin_of_safety(self):
        estimate = IntrinsicValueEstimate(
            ticker="AAPL",
            method="graham_number",
            intrinsic_value_per_share=120.0,
            current_price=187.0,
            margin_of_safety_pct=-55.8,
        )
        assert estimate.margin_of_safety_pct < 0

    def test_valid_methods(self):
        for method in ["dcf", "graham_number", "owner_earnings"]:
            estimate = IntrinsicValueEstimate(
                ticker="TEST",
                method=method,
                intrinsic_value_per_share=100.0,
                current_price=80.0,
                margin_of_safety_pct=20.0,
            )
            assert estimate.method == method

    def test_invalid_method_rejected(self):
        with pytest.raises(Exception):
            IntrinsicValueEstimate(
                ticker="TEST",
                method="invalid_method",
                intrinsic_value_per_share=100.0,
                current_price=80.0,
                margin_of_safety_pct=20.0,
            )


class TestMoatAssessment:
    def test_wide_moat(self):
        moat = MoatAssessment(
            ticker="AAPL",
            moat_rating="wide",
            moat_sources=["brand", "switching_costs", "network_effects"],
            gross_margin_stability=0.008,
            avg_competitive_advantage_gap=0.123,
            summary="Strong ecosystem lock-in.",
        )
        assert moat.moat_rating == "wide"
        assert len(moat.moat_sources) == 3

    def test_no_moat(self):
        moat = MoatAssessment(
            ticker="XYZ",
            moat_rating="none",
            moat_sources=[],
            gross_margin_stability=0.15,
            summary="Commodity business with no structural advantages.",
        )
        assert moat.moat_rating == "none"
        assert moat.avg_competitive_advantage_gap is None

    def test_invalid_rating_rejected(self):
        with pytest.raises(Exception):
            MoatAssessment(
                ticker="TEST",
                moat_rating="ultra_wide",
                moat_sources=[],
                gross_margin_stability=0.01,
            )


class TestEarningsConsistency:
    def test_predictable_earnings(self):
        ec = EarningsConsistency(
            ticker="AAPL",
            eps_history=[2.31, 2.08, 2.30, 2.98, 2.97, 3.28, 5.61, 6.11, 6.13, 6.42],
            years=list(range(2015, 2025)),
            earnings_growth_cagr=0.142,
            positive_earnings_streak=10,
            coefficient_of_variation=0.18,
            is_predictable=True,
        )
        assert ec.is_predictable is True
        assert ec.positive_earnings_streak == 10
        assert len(ec.eps_history) == 10

    def test_unpredictable_earnings(self):
        ec = EarningsConsistency(
            ticker="CYCL",
            eps_history=[1.5, -0.3, 2.1, -1.0, 0.8],
            years=[2020, 2021, 2022, 2023, 2024],
            positive_earnings_streak=1,
            coefficient_of_variation=0.95,
            is_predictable=False,
        )
        assert ec.is_predictable is False

    def test_none_eps_values(self):
        ec = EarningsConsistency(
            ticker="NEW",
            eps_history=[None, None, 1.0, 1.5, 2.0],
            years=[2020, 2021, 2022, 2023, 2024],
            positive_earnings_streak=3,
            is_predictable=False,
        )
        assert ec.eps_history[0] is None


class TestDebtAnalysis:
    def test_conservative_debt(self):
        debt = DebtAnalysis(
            ticker="AAPL",
            debt_to_equity=0.3,
            current_ratio=2.5,
            interest_coverage=12.0,
            rating="conservative",
        )
        assert debt.rating == "conservative"

    def test_dangerous_debt(self):
        debt = DebtAnalysis(
            ticker="RISK",
            debt_to_equity=4.0,
            current_ratio=0.6,
            interest_coverage=1.2,
            rating="dangerous",
        )
        assert debt.rating == "dangerous"

    def test_default_rating(self):
        debt = DebtAnalysis(ticker="TEST")
        assert debt.rating == "moderate"

    def test_none_fields(self):
        debt = DebtAnalysis(ticker="NEW", rating="moderate")
        assert debt.debt_to_equity is None
        assert debt.current_ratio is None
        assert debt.interest_coverage is None


class TestValueScorecard:
    def test_full_scorecard(self):
        scorecard = ValueScorecard(
            ticker="AAPL",
            company_name="Apple Inc.",
            current_price=187.44,
            intrinsic_values=[
                IntrinsicValueEstimate(
                    ticker="AAPL",
                    method="dcf",
                    intrinsic_value_per_share=158.20,
                    current_price=187.44,
                    margin_of_safety_pct=-15.6,
                ),
                IntrinsicValueEstimate(
                    ticker="AAPL",
                    method="graham_number",
                    intrinsic_value_per_share=121.40,
                    current_price=187.44,
                    margin_of_safety_pct=-35.2,
                ),
            ],
            moat=MoatAssessment(
                ticker="AAPL",
                moat_rating="wide",
                moat_sources=["brand", "switching_costs"],
                gross_margin_stability=0.008,
                summary="Wide moat.",
            ),
            earnings=EarningsConsistency(
                ticker="AAPL",
                eps_history=[6.42],
                years=[2024],
                is_predictable=True,
                positive_earnings_streak=10,
            ),
            debt=DebtAnalysis(ticker="AAPL", debt_to_equity=1.73, rating="moderate"),
            overall_rating="hold",
            verdict="Wide moat but overvalued.",
            key_strengths=["Wide moat", "Predictable earnings"],
            key_risks=["Overvalued", "China exposure"],
        )
        assert scorecard.overall_rating == "hold"
        assert len(scorecard.intrinsic_values) == 2
        assert scorecard.moat.moat_rating == "wide"

    def test_minimal_scorecard(self):
        scorecard = ValueScorecard(
            ticker="TEST",
            company_name="Test Corp",
            current_price=50.0,
        )
        assert scorecard.overall_rating == "hold"
        assert scorecard.intrinsic_values == []
        assert scorecard.moat is None

    def test_valid_ratings(self):
        for rating in ["strong_buy", "buy", "hold", "avoid"]:
            scorecard = ValueScorecard(
                ticker="TEST",
                company_name="Test",
                current_price=100.0,
                overall_rating=rating,
            )
            assert scorecard.overall_rating == rating

    def test_invalid_rating_rejected(self):
        with pytest.raises(Exception):
            ValueScorecard(
                ticker="TEST",
                company_name="Test",
                current_price=100.0,
                overall_rating="sell",
            )
