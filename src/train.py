import numpy as np
from src.models import make_regression_pipeline
from src.evaluation import regression_metrics

def main():
    # Fake data just to test the pipeline
    X = np.random.randn(100, 3)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100)

    model = make_regression_pipeline()
    model.fit(X, y)
    preds = model.predict(X)

    metrics = regression_metrics(y, preds)
    print("Training metrics:", metrics)

if __name__ == "__main__":
    main()
