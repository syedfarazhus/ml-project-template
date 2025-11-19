from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MSE": mse, "RMSE": rmse}

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc}
