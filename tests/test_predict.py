"""
Tests for the updated FlightPredictor (LightGBM v3).
Run from MTP root: pytest tests/test_predict.py -v
Requires trained model artifacts in model/ (run train_lgbm.py first).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from predict import FlightPredictor

# A high-delay-risk flight input
DELAYED_FEATURES = {
    "CRS_ELAPSED_TIME": 150, "DISTANCE": 800,
    "origin_humidity": 85, "dest_humidity": 80,
    "OP_CARRIER_Southwest Airlines": 1,
    "origin_city_new york": 1,
    "destination_city_miami": 1,
    "Season_Summer": 1, "WeekendFlagEncoded": 1,
    "CRS_DEP_TIME_4": 1, "MONTH_7": 1,
}
# A low-delay-risk flight input
ONTIME_FEATURES = {
    "CRS_ELAPSED_TIME": 90, "DISTANCE": 400,
    "origin_humidity": 40, "dest_humidity": 35,
    "OP_CARRIER_Delta Airlines": 1,
    "origin_city_atlanta": 1,
    "Season_Autumn": 1, "WeekendFlagEncoded": 0,
    "CRS_DEP_TIME_2": 1, "MONTH_9": 1,
    "origin_condition_text_Sunny": 1,
}

@pytest.fixture(scope="module")
def predictor():
    return FlightPredictor()


def test_predict_returns_all_required_keys(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    required = {
        "delayed", "probability", "probability_pct",
        "cluster", "model_used", "confidence",
        "expected_delay_min", "delay_range",
        "delay_category", "delay_breakdown",
    }
    missing = required - result.keys()
    assert not missing, f"Response missing keys: {missing}"


def test_probability_in_unit_interval(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    assert 0.0 <= result["probability"] <= 1.0


def test_model_used_is_lgbm(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    assert result["model_used"] == "lgbm", (
        f"Expected 'lgbm', got '{result['model_used']}'"
    )


def test_cluster_is_minus_one(predictor):
    """cluster field should be -1 (no clustering in new pipeline)."""
    result = predictor.predict(DELAYED_FEATURES)
    assert result["cluster"] == -1


def test_ontime_flight_has_zero_delay(predictor):
    result = predictor.predict(ONTIME_FEATURES)
    if not result["delayed"]:
        assert result["expected_delay_min"] == 0
        assert result["delay_category"] == "on-time"
        assert result["delay_range"] == "No delay expected"


def test_delayed_result_has_positive_delay_min(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        assert result["expected_delay_min"] > 0
        assert result["delay_category"] in ("minor", "moderate", "significant", "severe")


def test_breakdown_has_correct_keys(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        assert set(result["delay_breakdown"].keys()) == {
            "carrier", "weather", "nas", "late_aircraft"
        }


def test_breakdown_sums_near_expected_delay(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        total = sum(result["delay_breakdown"].values())
        expected = result["expected_delay_min"]
        assert abs(total - expected) <= 2, (
            f"Breakdown sum {total} differs from expected_delay_min {expected} by >2 min"
        )


def test_delay_range_contains_dash_when_delayed(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        assert "–" in result["delay_range"], (
            f"delay_range should contain en-dash, got: {result['delay_range']}"
        )


def test_predict_batch_returns_correct_count(predictor):
    results = predictor.predict_batch([DELAYED_FEATURES, ONTIME_FEATURES])
    assert len(results) == 2
    assert all("delayed" in r for r in results)
