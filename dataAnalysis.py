import pandas as pd
import csv

# Read the CSV file into a DataFrame
class dataSample:
    type = "" # s(simple)/m(memory)/t(target)/f(memory+target)
    size = 0
    h = 0
    w = 0
    mean_time = 0
    mean_samples = 0

dataSamples = []

with open('DQN.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
     dataSamples.append(dataSample(type="s",h = int(row[0]),w = int(row[1])))




# Access data in the DataFrame using column names or indexing
print(df.iloc[0])  