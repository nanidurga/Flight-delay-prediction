"""
Tests for POST /feedback endpoint and updated /model/info.
Run from MTP root: pytest tests/test_api_feedback.py -v
"""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import api.main as main_module
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def patch_feedback_path(tmp_path, monkeypatch):
    """Redirect feedback writes to a temp file so tests don't pollute data/feedback.csv."""
    fb = tmp_path / "feedback.csv"
    fb.write_text("flight_id,timestamp,actual_delayed,actual_delay_min\n")
    monkeypatch.setattr(main_module, "FEEDBACK_PATH", fb)
    return fb


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_feedback_valid_returns_200(client):
    resp = client.post("/feedback", json={
        "flight_id": "UA123_2026-04-12",
        "actual_delayed": True,
        "actual_delay_min": 87,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recorded"
    assert isinstance(body["feedback_count"], int)
    assert body["feedback_count"] >= 1


def test_feedback_duplicate_flight_id_ignored(client, tmp_path, monkeypatch):
    fb = tmp_path / "fb_dup.csv"
    fb.write_text("flight_id,timestamp,actual_delayed,actual_delay_min\n")
    monkeypatch.setattr(main_module, "FEEDBACK_PATH", fb)

    payload = {"flight_id": "DUP001", "actual_delayed": True, "actual_delay_min": 45}
    client.post("/feedback", json=payload)
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200

    rows = [r for r in fb.read_text().strip().split("\n")[1:] if r.startswith("DUP001")]
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"


def test_feedback_delay_min_over_800_rejected(client):
    resp = client.post("/feedback", json={
        "flight_id": "BAD001", "actual_delayed": True, "actual_delay_min": 801,
    })
    assert resp.status_code == 422


def test_feedback_delayed_true_but_zero_minutes_rejected(client):
    resp = client.post("/feedback", json={
        "flight_id": "BAD002", "actual_delayed": True, "actual_delay_min": 0,
    })
    assert resp.status_code == 422


def test_feedback_delayed_false_but_high_minutes_rejected(client):
    resp = client.post("/feedback", json={
        "flight_id": "BAD003", "actual_delayed": False, "actual_delay_min": 60,
    })
    assert resp.status_code == 422


def test_model_info_has_new_keys(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_version", "roc_auc", "model_type",
                "regression_mae", "incremental_updates"):
        assert key in body, f"Missing key: {key}"
