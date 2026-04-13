"""
predictor.py — Singleton wrapper around the trained model.
Loaded once at startup so every API request is fast.
"""

import sys
from pathlib import Path

# Allow importing predict.py from the project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from predict import FlightPredictor

_predictor: FlightPredictor | None = None


def get_predictor() -> FlightPredictor:
    global _predictor
    if _predictor is None:
        _predictor = FlightPredictor()
    return _predictor
