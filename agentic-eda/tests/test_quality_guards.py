"""Regression tests for quality guards in profiling and visualization."""

from __future__ import annotations

import unittest

import pandas as pd

from src.profiler import profile_dataset
from src.visualization_recommender import recommend_visualizations


class QualityGuardTests(unittest.TestCase):
    def test_target_detection_prefers_price_over_ambiguous_y(self) -> None:
        df = pd.DataFrame(
            {
                "carat": [0.3, 0.4, 0.5, 1.0, 1.1, 0.8],
                "x": [4.3, 4.6, 4.9, 6.5, 6.7, 6.1],
                "y": [4.4, 4.7, 5.0, 6.6, 6.8, 6.2],
                "price": [350, 420, 600, 2400, 2700, 1900],
                "cut": ["Ideal", "Premium", "Good", "Ideal", "Premium", "Good"],
            }
        )

        profile = profile_dataset(df, "diamonds_like")
        self.assertEqual(profile["likely_target_col"], "price")

    def test_recommender_excludes_identifier_like_columns(self) -> None:
        rows = 40
        df = pd.DataFrame(
            {
                "PassengerId": list(range(1, rows + 1)),
                "Name": [f"Passenger_{i}" for i in range(rows)],
                "Sex": ["male" if i % 2 == 0 else "female" for i in range(rows)],
                "Pclass": [1 if i % 3 == 0 else 3 for i in range(rows)],
                "Fare": [10.0 + (i % 7) * 5 for i in range(rows)],
                "Survived": [1 if i % 4 == 0 else 0 for i in range(rows)],
            }
        )

        profile = profile_dataset(df, "titanic_like")
        analysis_results = {
            "missing_value_analysis": {"columns_with_missing": []},
            "correlation_analysis": {
                "strongest_pair": {"feature_1": "Pclass", "feature_2": "Fare", "correlation": -0.4}
            },
        }
        recommendations = recommend_visualizations(df, profile, analysis_results)["recommendations"]

        x_columns = {str(rec.get("x")) for rec in recommendations}
        self.assertNotIn("PassengerId", x_columns)
        self.assertNotIn("Name", x_columns)


if __name__ == "__main__":
    unittest.main()
