import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('DQN.csv')

# Access data in the DataFrame using column names or indexing
print(df['time'])
print(df.iloc[0])  