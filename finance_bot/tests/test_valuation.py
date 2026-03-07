"""Tests for ValuationAgent."""

import pytest

from finance_bot.valuation import ValuationAgent


@pytest.fixture
def agent():
    return ValuationAgent(verbose=False)


@pytest.fixture
def sample_data():
    """Realistic DataAgent output for AAPL."""
    return {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "current_price": 187.44,
        "metrics": {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "eps": 6.42,
            "pe_ratio": 29.3,
            "profit_margin": 0.263,
            "debt_to_equity": 173.0,
            "current_ratio": 1.07,
            "revenue_growth": 0.05,
        },
        "eps_history": {
            "years": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "eps": [2.31, 2.08, 2.30, 2.98, 2.97, 3.28, 5.61, 6.11, 6.13, 6.42],
        },
        "free_cash_flow": {
            "years": [2021, 2022, 2023, 2024],
            "fcf": [93000000000, 111400000000, 99600000000, 108100000000],
        },
        "book_value_per_share": 4.38,
        "shares_outstanding": 15400000000,
        "interest_coverage": 30.44,
        "owner_earnings": {
            "net_income": 100800000000,
            "depreciation_amortization": 11400000000,
            "avg_capex_3yr": 10533333333.33,
            "owner_earnings": 101666666666.67,
        },
        "margins": {
            "gross_profit_margins": [0.438, 0.433, 0.441, 0.462],
            "competitive_advantage_gaps": [0.13, 0.15, 0.17, 0.18],
        },
    }


@pytest.fixture
def minimal_data():
    """Minimal data with missing fields."""
    return {
        "ticker": "NEW",
        "company_name": "New Corp",
        "current_price": 50.0,
        "metrics": {},
        "eps_history": {"years": [], "eps": []},
        "free_cash_flow": {"years": [], "fcf": []},
        "book_value_per_share": None,
        "shares_outstanding": None,
        "interest_coverage": None,
        "owner_earnings": {},
        "margins": None,
    }


class TestDCF:
    def test_dcf_produces_value(self, agent, sample_data):
        result = agent.dcf_valuation(sample_data)
        assert result is not None
        assert result.method == "dcf"
        assert result.intrinsic_value_per_share > 0
        assert result.current_price == 187.44

    def test_dcf_caps_growth_at_8pct(self, agent, sample_data):
        result = agent.dcf_valuation(sample_data)
        assert result.assumptions["growth_rate"] <= 0.08

    def test_dcf_no_fcf(self, agent, minimal_data):
        result = agent.dcf_valuation(minimal_data)
        assert result is None

    def test_dcf_single_fcf(self, agent, sample_data):
        sample_data["free_cash_flow"]["fcf"] = [100000000000]
        result = agent.dcf_valuation(sample_data)
        assert result is not None
        assert result.assumptions["growth_rate"] == 0.03  # Default


class TestGrahamNumber:
    def test_graham_positive(self, agent, sample_data):
        result = agent.graham_number(sample_data)
        assert result is not None
        assert result.method == "graham_number"
        # sqrt(22.5 * 6.42 * 4.38) ≈ 25.16
        assert result.intrinsic_value_per_share == pytest.approx(25.16, rel=0.01)

    def test_graham_negative_eps(self, agent, sample_data):
        sample_data["metrics"]["eps"] = -1.0
        result = agent.graham_number(sample_data)
        assert result is None

    def test_graham_negative_bvps(self, agent, sample_data):
        sample_data["book_value_per_share"] = -5.0
        result = agent.graham_number(sample_data)
        assert result is None

    def test_graham_missing_data(self, agent, minimal_data):
        result = agent.graham_number(minimal_data)
        assert result is None


class TestOwnerEarnings:
    def test_owner_earnings_value(self, agent, sample_data):
        result = agent.owner_earnings_valuation(sample_data)
        assert result is not None
        assert result.method == "owner_earnings"
        assert result.intrinsic_value_per_share > 0

    def test_owner_earnings_missing(self, agent, minimal_data):
        result = agent.owner_earnings_valuation(minimal_data)
        assert result is None

    def test_owner_earnings_negative(self, agent, sample_data):
        sample_data["owner_earnings"]["owner_earnings"] = -5000
        result = agent.owner_earnings_valuation(sample_data)
        assert result is None


class TestMoatAssessment:
    def test_wide_moat(self, agent, sample_data):
        result = agent.assess_moat(sample_data)
        assert result is not None
        assert result.moat_rating == "wide"
        assert result.gross_margin_stability < 0.03

    def test_no_moat_data(self, agent, minimal_data):
        result = agent.assess_moat(minimal_data)
        assert result is None

    def test_volatile_margins_no_moat(self, agent, sample_data):
        sample_data["margins"]["gross_profit_margins"] = [0.20, 0.35, 0.15, 0.40]
        sample_data["margins"]["competitive_advantage_gaps"] = [-0.05, 0.02, -0.10, 0.01]
        result = agent.assess_moat(sample_data)
        assert result is not None
        assert result.moat_rating == "none"


class TestEarningsConsistency:
    def test_predictable(self, agent, sample_data):
        result = agent.assess_earnings_consistency(sample_data)
        assert result is not None
        assert result.is_predictable is True
        assert result.positive_earnings_streak == 10
        assert result.earnings_growth_cagr is not None

    def test_empty_history(self, agent, minimal_data):
        result = agent.assess_earnings_consistency(minimal_data)
        assert result is None

    def test_volatile_earnings(self, agent, sample_data):
        sample_data["eps_history"]["eps"] = [1.0, -2.0, 3.0, -1.0, 0.5]
        sample_data["eps_history"]["years"] = [2020, 2021, 2022, 2023, 2024]
        result = agent.assess_earnings_consistency(sample_data)
        assert result is not None
        assert result.is_predictable is False


class TestDebtAnalysis:
    def test_moderate_debt(self, agent, sample_data):
        result = agent.assess_debt(sample_data)
        assert result is not None
        # D/E 173 -> 1.73 after conversion
        assert result.debt_to_equity == 1.73
        assert result.rating in ("aggressive", "moderate")

    def test_no_debt_data(self, agent, minimal_data):
        result = agent.assess_debt(minimal_data)
        assert result is not None
        assert result.rating == "moderate"  # Default

    def test_high_de_conversion(self, agent, sample_data):
        """yfinance returns D/E as percentage; should convert."""
        sample_data["metrics"]["debt_to_equity"] = 250.0
        result = agent.assess_debt(sample_data)
        assert result.debt_to_equity == 2.50


class TestScorecard:
    def test_full_scorecard(self, agent, sample_data):
        scorecard = agent.compute_scorecard(sample_data)
        assert scorecard.ticker == "AAPL"
        assert scorecard.current_price == 187.44
        assert len(scorecard.intrinsic_values) >= 1
        assert scorecard.overall_rating in ("strong_buy", "buy", "hold", "avoid")
        assert scorecard.verdict != ""

    def test_minimal_scorecard(self, agent, minimal_data):
        scorecard = agent.compute_scorecard(minimal_data)
        assert scorecard.ticker == "NEW"
        assert scorecard.overall_rating == "hold"
        assert len(scorecard.intrinsic_values) == 0

    def test_scorecard_has_strengths_risks(self, agent, sample_data):
        scorecard = agent.compute_scorecard(sample_data)
        assert isinstance(scorecard.key_strengths, list)
        assert isinstance(scorecard.key_risks, list)

    def test_scorecard_serializable(self, agent, sample_data):
        scorecard = agent.compute_scorecard(sample_data)
        json_str = scorecard.model_dump_json()
        assert "AAPL" in json_str
