import pandas as pd
import yaml

class RemoveOutlierClass:
    def __init__(self, file_path: str, yaml_path: str):
        # Read the CSV file into a dataframe
        self.df = pd.read_csv(file_path)
        self.yaml_path= yaml_path
        # Load the airlineName configurations from the YAML file
        # self.airlineName = self.load_yaml(yaml_path)
        # Initialize df as empty with the same columns
        self.cleaned_df = pd.DataFrame(columns=list(self.df.columns))

    def load_yaml(self):
        with open(self.yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            print(data['airlineName'])
            # Assign the 'airlineName' from the loaded YAML data
            return data['airlineName']  # If not found, return an empty dictionary


    def remove_outliers(self,airlineName):
        print(airlineName)
        for airline, quartiles in airlineName.items():
            # Filter the data for the specific airline
            airDataSet = self.df[self.df['Airline'] == airline]
            
            # Calculate the IQR for the 'Price' column
            q1 = airDataSet['Price'].quantile(quartiles[0])
            q3 = airDataSet['Price'].quantile(quartiles[1])
            IQR = q3 - q1
            lowerLimit = q1 - IQR * 1.5
            upperLimit = q3 + IQR * 1.5

            # Find the outliers based on the limits
            lowerLimitIndex = airDataSet[airDataSet['Price'] <= lowerLimit].index
            upperLimitIndex = airDataSet[airDataSet['Price'] >= upperLimit].index

            # Remove outliers only if the dataset has more than 5 rows
            if airDataSet.shape[0] > 5:
                airDataSet.drop(lowerLimitIndex, axis=0, inplace=True)
                airDataSet.drop(upperLimitIndex, axis=0, inplace=True)

            # Append the cleaned dataset for this airline
            self.cleaned_df = pd.concat([self.cleaned_df, airDataSet], axis=0)
            print(self.cleaned_df)
        
        # Return the cleaned dataframe after removing outliers
        return self.cleaned_df
    
    def save_file(self, noOutlierDataFilePath):
        # Save the cleaned dataframe to a CSV file
        self.cleaned_df.to_csv(noOutlierDataFilePath, index=False)

# Example usage:
if __name__ == "__main__":
    file_path = "Data\\02_CleanedData\\CleanedData.csv"  # Path to the cleaned CSV file
    yaml_path = "contants.yaml"  # Path to the YAML file containing quartile data for airlines
    noOutlierDataFilePath = "Data\\03_noOutlierData\\noOutlierDataFile.csv"  # Path to save cleaned data
    
    # Create an instance of RemoveOutlierClass
    RemoveOutlierObj = RemoveOutlierClass(file_path, yaml_path)
    # Remove outliers from the data
    airlineName = RemoveOutlierObj.load_yaml()
    RemoveOutlierObj.remove_outliers(airlineName)
    # Save the cleaned data to a new CSV file
    RemoveOutlierObj.save_file(noOutlierDataFilePath)
