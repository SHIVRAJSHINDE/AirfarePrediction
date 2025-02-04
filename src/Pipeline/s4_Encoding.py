import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

class EncodingAndScalingClass:
    def __init__(self, file_path: str):
        # Read the dataset from the file
        self.df = pd.read_csv(file_path)
        # Remove 'Unnamed: 0' column if exists (usually index from CSV export)
        self.df = self.df.drop('Unnamed: 0', axis=1, errors='ignore')
        # Initialize train/test split
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipe = self.encoding_and_scaling()

    def read_file(self):
        # Drop the 'Unnamed: 0' column (if present) and return the cleaned DataFrame
        self.df = self.df.drop('Unnamed: 0', axis=1, errors='ignore')
        return self.df

    def split_df_to_X_y(self):
        # Split the dataframe into X (features) and y (target)
        X = self.df.drop(['Price'], axis=1)
        y = self.df['Price']
        return X, y

    def train_test_split(self, X, y):
        # Split the data into training and testing sets (80%/20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def encoding_and_scaling(self):
        # Define transformers for encoding and scaling
        trf1 = ColumnTransformer([
            ('OneHot', OneHotEncoder(drop='first', handle_unknown='ignore'), [0, 1, 2])
        ], remainder='passthrough')

        trf2 = ColumnTransformer([
            ('Ordinal', OrdinalEncoder(categories=[['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']]), [16])
        ], remainder='passthrough')

        trf3 = ColumnTransformer([
            ('scale', StandardScaler(), slice(25))  # Assuming that columns beyond index 25 are numeric
        ])
        
        # Create pipeline to combine the transformers
        pipe = make_pipeline(trf1, trf2, trf3)
        return pipe

    def save_X_train(self):
        # Apply the pipeline to X_train and save it as CSV
        X_train_transformed = self.pipe.fit_transform(self.X_train)
        X_train_transformed = pd.DataFrame(X_train_transformed)
        X_train_transformed.to_csv("Data\\04_encoded_Data\\X_train.csv", index=False)

    def save_X_test(self):
        # Apply the pipeline to X_test and save it as CSV
        X_test_transformed = self.pipe.transform(self.X_test)
        X_test_transformed = pd.DataFrame(X_test_transformed)
        X_test_transformed.to_csv("Data\\04_encoded_Data\\X_test.csv", index=False)

    def save_y_train(self):
        # Save y_train as CSV
        self.y_train.to_csv("Data\\04_encoded_Data\\y_train.csv", index=False)

    def save_y_test(self):
        # Save y_test as CSV
        self.y_test.to_csv("Data\\04_encoded_Data\\y_test.csv", index=False)

# Example usage:
if __name__ == "__main__":
    file_path = "Data\\03_noOutlierData\\noOutlierDataFile.csv"  # Path to the input file
    encoding_and_scaling_obj = EncodingAndScalingClass(file_path)

    # Step 1: Read and clean the file
    encoding_and_scaling_obj.read_file()

    # Step 2: Split the data into X (features) and y (target)
    X, y = encoding_and_scaling_obj.split_df_to_X_y()

    # Step 3: Split data into training and testing sets
    encoding_and_scaling_obj.train_test_split(X, y)

    # Step 4: Apply encoding and scaling and save the datasets
    encoding_and_scaling_obj.save_X_train()
    encoding_and_scaling_obj.save_X_test()
    encoding_and_scaling_obj.save_y_train()
    encoding_and_scaling_obj.save_y_test()
