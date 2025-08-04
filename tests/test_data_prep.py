import json

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (build_preprocessor, load_data, prepare_data,
                           save_baseline_stats)


def test_load_data(tmp_path):
    csv = tmp_path / "data.csv"
    df_in = pd.DataFrame(
        {
            "id": [1, 2],
            "age": [30, 40],
            "avg_glucose_level": [85.0, 90.0],
            "bmi": [22.2, 27.5],
            "gender": ["Male", "Female"],
            "hypertension": [0, 1],
            "heart_disease": [0, 0],
            "ever_married": ["No", "Yes"],
            "work_type": ["Private", "Self-employed"],
            "Residence_type": ["Urban", "Rural"],
            "smoking_status": ["never smoked", "formerly smoked"],
            "stroke": [0, 1],
        }
    )
    df_in.to_csv(csv, index=False)
    df = load_data(str(csv))
    pd.testing.assert_frame_equal(df_in, df)


def test_prepare_data():
    df = pd.DataFrame(
        {
            "age": list(range(100)),
            "avg_glucose_level": np.linspace(60, 200, 100),
            "bmi": np.linspace(18, 40, 100),
            "gender": ["Male"] * 100,
            "hypertension": [0] * 50 + [1] * 50,
            "heart_disease": [0] * 50 + [1] * 50,
            "ever_married": ["Yes"] * 100,
            "work_type": ["Private"] * 100,
            "Residence_type": ["Urban"] * 100,
            "smoking_status": ["never smoked"] * 100,
            "stroke": [0] * 80 + [1] * 20,
        }
    )
    X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, random_state=0)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert set(y_train.unique()).issubset({0, 1})


def test_build_preprocessor_and_transform():
    df = pd.DataFrame(
        {
            "age": [25, 35],
            "avg_glucose_level": [80.0, 90.0],
            "bmi": [21.5, 26.3],
            "gender": ["Male", "Female"],
            "ever_married": ["No", "Yes"],
            "work_type": ["Private", "Govt_job"],
            "Residence_type": ["Urban", "Rural"],
            "smoking_status": ["never smoked", "formerly smoked"],
        }
    )
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    # Expect numeric + one-hot columns
    assert X.shape[0] == 2
    assert X.shape[1] > df.shape[1]


def test_save_baseline_stats(tmp_path):
    df = pd.DataFrame(
        {
            "age": [30, 50, 70],
            "avg_glucose_level": [100.0, 110.0, 120.0],
            "bmi": [20.0, 25.0, 30.0],
        }
    )
    out = tmp_path / "interim" / "baseline_stats.json"
    save_baseline_stats(df, path=str(out))
    assert out.exists()
    stats = json.loads(out.read_text())
    assert pytest.approx(stats["age"]) == 50
    assert pytest.approx(stats["avg_glucose_level"]) == 110.0
    assert pytest.approx(stats["bmi"]) == 25.0
