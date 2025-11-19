from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

def make_regression_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression())
    ])

def make_classification_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
