from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data import make_synthetic_regression
from src.models import make_regression_pipeline
from src.evaluation import regression_report


MODELS_DIR = Path("models")


def main():
    # 1) Prepare data
    df = make_synthetic_regression()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2) Build model
    pipeline = make_regression_pipeline()

    # 3) Train
    pipeline.fit(X_train, y_train)

    # 4) Evaluate
    preds = pipeline.predict(X_test)
    metrics = regression_report(y_test, preds)
    print("Training complete. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5) (Optional) Save model later – once you’re ready
    # from joblib import dump
    # MODELS_DIR.mkdir(exist_ok=True)
    # dump(pipeline, MODELS_DIR / "regression_model.pkl")


if __name__ == "__main__":
    main()
