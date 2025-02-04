import pandas as pd

class DataCleaningClass:
    def __init__(self, file_path: str):
        """Initializes the DataCleaningClass with the given CSV file path."""
        self.df = pd.read_csv(file_path)

    def clean_total_stops(self):
        mode_of_total_stops = self.df['Total_Stops'].mode()[0]
        self.df['Total_Stops'].fillna(mode_of_total_stops, inplace=True)
        print("Missing values in 'Total_Stops' filled with mode.")
    
    def clean_airline_column(self):
        self.df['Airline'].replace("Multiple carriers Premium economy", "Multiple carriers", inplace=True)
        self.df['Airline'].replace("Jet Airways Business", "Jet Airways", inplace=True)
        self.df['Airline'].replace("Vistara Premium economy", "Vistara", inplace=True)
        print("Airline names cleaned.")

    def clean_destination_column(self):
        """Replace 'New Delhi' with 'Delhi' in the 'Destination' column."""
        self.df['Destination'].replace(to_replace="New Delhi", value="Delhi", inplace=True)
        print("'New Delhi' replaced with 'Delhi' in 'Destination'.")

    def create_duration_column(self):
        """Create a new column 'hoursMinutes' to represent flight duration in minutes."""
        self.df["hoursMinutes"] = 0
        for i in self.df.index:
            if " " in self.df.loc[i, 'Duration']:
                column1 = self.df.loc[i, 'Duration'].split(" ")[0]
                column2 = self.df.loc[i, 'Duration'].split(" ")[1]

                if "h" in column1:
                    column1 = (int(column1.replace("h", "")) * 60)
                elif "m" in column1:
                    column1 = (int(column1.replace("m", "")))

                if "h" in column2:
                    column2 = (int(column2.replace("h", "")) * 60)
                elif "m" in column2:
                    column2 = (int(column2.replace("m", "")))

                self.df.loc[i, 'hoursMinutes'] = column1 + column2
            else:
                column1 = self.df.loc[i, 'Duration']

                if "h" in column1:
                    column1 = (int(column1.replace("h", "")) * 60)
                elif "m" in column1:
                    column1 = (int(column1.replace("m", "")))

                self.df.loc[i, 'hoursMinutes'] = column1

        print("'hoursMinutes' column created from 'Duration'.")


    def process_date_time_columns(self):
        """Extracts and processes 'Date_of_Journey', 'Dep_Time', and 'Arrival_Time' columns."""
        self.df['Day'] = pd.to_datetime(self.df["Date_of_Journey"], format="%d-%m-%Y").dt.day
        self.df['Month'] = pd.to_datetime(self.df['Date_of_Journey'], format="%d-%m-%Y").dt.month
        self.df['Year'] = pd.to_datetime(self.df['Date_of_Journey'], format="%d-%m-%Y").dt.year

        self.df['Dept_Hour'] = pd.to_datetime(self.df['Dep_Time']).dt.hour
        self.df['Dept_Minute'] = pd.to_datetime(self.df['Dep_Time']).dt.minute

        self.df['Arr_Hour'] = pd.to_datetime(self.df['Arrival_Time']).dt.hour
        self.df['Arr_Minute'] = pd.to_datetime(self.df['Arrival_Time']).dt.minute

        print("Date and time columns processed.")

    def drop_unnecessary_columns(self):
        """Drops unnecessary columns from the DataFrame."""
        self.df = self.df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 'Route', 'Additional_Info'], axis=1)
        print("Unnecessary columns dropped.")

    def reorder_columns(self):
        """Reorders columns for the final DataFrame."""
        self.df = self.df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Day', 'Month', 'Year', 
                           'Dept_Hour', 'Dept_Minute', 'Arr_Hour', 'Arr_Minute', 'hoursMinutes', 'Price']]
        print("Columns reordered.")

    def save_data(self, cleaned_data_path):
        """Saves the cleaned data to a specified path."""
        self.df.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned data saved to: {cleaned_data_path}")



# Example usage:
if __name__ == "__main__":
    file_path = "Data\\01_RawData\\Airline.csv"
    cleaned_data_path = "Data\\02_CleanedData\\CleanedData.csv"
    
    # Create an instance of DataCleaningClass
    data_cleaning_obj = DataCleaningClass(file_path)

    # Apply the cleaning functions
    data_cleaning_obj.clean_total_stops()
    data_cleaning_obj.clean_airline_column()
    data_cleaning_obj.clean_destination_column()
    data_cleaning_obj.create_duration_column()

    # Process date and time related columns
    data_cleaning_obj.process_date_time_columns()

    # Drop unnecessary columns and reorder the remaining ones
    data_cleaning_obj.drop_unnecessary_columns()
    data_cleaning_obj.reorder_columns()

    # Save the cleaned data
    data_cleaning_obj.save_data(cleaned_data_path)
