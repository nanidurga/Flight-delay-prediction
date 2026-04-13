"""Shared pytest fixtures for all MTP tests.
Run all tests from MTP root: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

SAMPLE_FEATURE_NAMES = [
    "CRS_ELAPSED_TIME", "DISTANCE",
    "origin_temperature_celsius", "origin_humidity",
    "dest_temperature_celsius", "dest_humidity",
    "DAY", "WeekendFlagEncoded", "DayBeforeWeekendEncoded",
    "Season_Autumn", "Season_Spring", "Season_Summer", "Season_Winter",
    "OP_CARRIER_American Airlines", "OP_CARRIER_Southwest Airlines",
    "OP_CARRIER_Delta Airlines",
    "origin_city_new york", "origin_city_atlanta", "origin_city_chicago",
    "destination_city_miami", "destination_city_chicago",
    "destination_city_new york",
    "origin_condition_text_Sunny", "origin_condition_text_Partly cloudy",
    "origin_condition_text_Heavy rain",
    "dest_condition_text_Sunny", "dest_condition_text_Partly cloudy",
    "CRS_DEP_TIME_2", "CRS_DEP_TIME_3", "CRS_DEP_TIME_4",
    "CRS_ARR_TIME_2", "CRS_ARR_TIME_4",
    "MONTH_6", "MONTH_7", "MONTH_9",
]


@pytest.fixture
def sample_X():
    """10-row DataFrame with one-hot groups activated per row."""
    n = 10
    np.random.seed(42)
    data = {col: np.zeros(n) for col in SAMPLE_FEATURE_NAMES}
    data["CRS_ELAPSED_TIME"] = np.random.uniform(60, 300, n)
    data["DISTANCE"] = np.random.uniform(200, 2000, n)
    data["origin_humidity"] = np.random.uniform(30, 90, n)
    data["dest_humidity"] = np.random.uniform(30, 90, n)

    carriers = ["OP_CARRIER_American Airlines", "OP_CARRIER_Southwest Airlines",
                "OP_CARRIER_Delta Airlines"]
    for i in range(n):
        data[carriers[i % 3]][i] = 1.0

    origins = ["origin_city_new york", "origin_city_atlanta", "origin_city_chicago"]
    for i in range(n):
        data[origins[i % 3]][i] = 1.0

    dests = ["destination_city_miami", "destination_city_chicago",
             "destination_city_new york"]
    for i in range(n):
        data[dests[i % 3]][i] = 1.0

    data["origin_condition_text_Sunny"][0] = 1.0
    data["origin_condition_text_Partly cloudy"][1] = 1.0
    data["origin_condition_text_Heavy rain"][2] = 1.0
    data["dest_condition_text_Sunny"][0] = 1.0
    data["dest_condition_text_Partly cloudy"][1] = 1.0

    for i in range(5):
        data["MONTH_7"][i] = 1.0
    for i in range(5, 10):
        data["MONTH_9"][i] = 1.0

    for i in range(3):
        data["CRS_DEP_TIME_2"][i] = 1.0
    for i in range(3, 6):
        data["CRS_DEP_TIME_3"][i] = 1.0
    for i in range(6, 10):
        data["CRS_DEP_TIME_4"][i] = 1.0

    return pd.DataFrame(data)


@pytest.fixture
def sample_y_total():
    return np.array([0, 120, 0, 45, 0, 80, 0, 200, 0, 60], dtype=float)


@pytest.fixture
def sample_y_status():
    return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
