import pandas as pd

class DataIngestionClass:
    def __init__(self, source_path: str, destination_path: str):
        self.source_path = source_path
        self.destination_path = destination_path

    def read_csv(self):
        df = pd.read_csv(self.source_path)
        print(df)
        return df

    def save_csv(self, df):
        df.to_csv(self.destination_path, index=False)  # index=False to avoid writing row indices


# This block will only execute if this script is run directly
if __name__ == "__main__":
    source_path = 'C:\\Users\\SHIVRAJ SHINDE\\JupiterWorking\\XL_ML\\Z_DataSets\\01_AirlineData\\Airline.csv'
    destination_path = "Data\\01_RawData\\Airline.csv"

    file_handler = DataIngestionClass(source_path, destination_path)

    df = file_handler.read_csv()
    file_handler.save_csv(df)
