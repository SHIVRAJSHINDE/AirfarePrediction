import yaml
import numpy as np
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet


class ModelTrainerClass:

    def __init__(self, X_train_path, y_train_path, params_path):
        """Initialize the trainer  with paths and load necessary data."""
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.params_path = params_path

    def load_X_train(self):
        """Load X_train from the provided file path."""
        X_train = pd.read_csv(self.X_train_path)
        self.X_train = np.array(X_train)
        return self.X_train

    def load_y_train(self):
        """Load y_train from the provided file path."""
        y_train = pd.read_csv(self.y_train_path)
        self.y_train = np.array(y_train).ravel()  # Ensure it's a flat array
        return self.y_train

    def load_params(self):
        """Load parameters from the YAML file."""
        with open(self.params_path, "r") as file:
            self.modelWithParams = yaml.safe_load(file)
            self.modelWithParams = self.modelWithParams['model']
            

        return self.modelWithParams
    
    def get_Model_class(self,model):
        model_class = globals()[model.split('.')[-1]]  # Get the class from the model name
        model = model_class()  # Instantiate the model
        return model

    def train_model(self, model, param_grid):
        """Train the model using RandomizedSearchCV."""
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=5)
        random_search.fit(self.X_train, self.y_train)

        best_model = random_search.best_estimator_
        best_params = random_search.best_params_

        best_model.fit(self.X_train, self.y_train)
        predicted_y = best_model.predict(self.X_train)

        return best_model, best_params, predicted_y

    def calculate_metrics(self, actual_X, actual_y, predicted_y):
        """Calculate evaluation metrics."""
        mse = mean_squared_error(actual_y, predicted_y)
        mae = mean_absolute_error(actual_y, predicted_y)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_y, predicted_y)

        # Adjusted R²
        n = len(actual_y)
        p = actual_X.shape[1]  # Use actual_X to get the number of features
        aR2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return mse, mae, rmse, r2, aR2

    def get_Model_Name(self, model):
        """Get the model name."""
        model_name = str(model.__class__.__name__)
        return model_name


class MLflowLoggerClass:

    def __init__(self, tracking_uri):
        """Initialize MLflowLogger with the tracking URI."""
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)

    def log_results(self, model_name, best_model, best_params, mse, mae, rmse, r2, aR2):
        """Log results to MLflow."""
        with mlflow.start_run(run_name=model_name):
            # Log metrics to MLflow
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("Adjusted R2", aR2)

            # Log best parameters
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"Best {param_name}", param_value)

            # Log the model
            mlflow.sklearn.log_model(best_model, f"{model_name}_model")

            # Print the metrics
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"RMSE: {rmse}")
            print(f"R2: {r2}")
            print(f"Adjusted R2: {aR2}")


# Main script execution
if __name__ == "__main__":

    X_train_path = "Data\\04_encoded_Data\\X_train.csv"
    y_train_path = "Data\\04_encoded_Data\\y_train.csv"
    params_path = "modelsParams.yaml"
    tracking_uri = "http://localhost:5000"

    # Initialize the ModelTrainerObj
    ModelTrainerObj = ModelTrainerClass(X_train_path, y_train_path, params_path)

    # Load data and params
    X_train = ModelTrainerObj.load_X_train()
    y_train = ModelTrainerObj.load_y_train()
    modelWithParams = ModelTrainerObj.load_params()
    print("----------------------------------------------------")
    #print(modelParams['model'])
    for value in modelWithParams.values():
        model = value['model']
        model = ModelTrainerObj.get_Model_class(model)
        params_Grid = value['param']

        print(model)
        print(params_Grid)

        # Train the model
        best_model, best_params, predicted_y = ModelTrainerObj.train_model(model, params_Grid)

        # Calculate metrics
        mse, mae, rmse, r2, aR2 = ModelTrainerObj.calculate_metrics(X_train, y_train, predicted_y)

        # Log the model name
        model_name = ModelTrainerObj.get_Model_Name(model)
        print(f"Model Name: {model_name}")

        # Initialize MLflowLogger and log results
        MLflowLoggerObj = MLflowLoggerClass(tracking_uri)  # Replace with your URI
        MLflowLoggerObj.log_results(model_name, best_model, best_params, mse, mae, rmse, r2, aR2)
