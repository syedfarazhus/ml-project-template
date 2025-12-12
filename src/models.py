from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def make_regression_pipeline(random_state: int = 42) -> Pipeline:
    """
    Returns a regression pipeline with scaling + RandomForest.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )