import numpy as np
import pandas as pd


def make_synthetic_regression(
    n_samples: int = 500,
    n_features: int = 5,
    noise: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Creates a synthetic regression dataset as a placeholder.
    Later you will replace this with a real dataset loader.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))

    # Simple linear relationship with some noise
    coef = np.array([3, -2, 1.5, 0.5, 0.2])[:n_features]
    y = X @ coef + noise * rng.normal(size=n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df