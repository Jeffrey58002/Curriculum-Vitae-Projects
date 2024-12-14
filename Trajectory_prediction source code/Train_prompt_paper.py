import pandas as pd

# Load the CSV file to check its content
data_path = '/mnt/data/shortened_pred.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
data.head(), data.info(), data.describe()
