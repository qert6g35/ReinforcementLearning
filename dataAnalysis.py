import pandas as pd
import matplotlib.pyplot as plt
import csv

# Read the CSV file into a DataFrame
# class dataSample:
#     type = "" # s(simple)/m(memory)/t(target)/f(memory+target)
#     size = 0
#     h = 0
#     w = 0
#     mean_time = 0
#     mean_iters = 0
#     samples = 0

# dataSamples = []

# with open('DQN.csv', 'r') as csvfile:
#   reader = csv.reader(csvfile)
#   for row in reader:
#       addDataSample = True
#       for sample in dataSamples:
#         if(sample.type == "s"):
#            sample.mean_time += row[]
#      dataSamples.append(dataSample(type="s",h = int(row[0]),w = int(row[1])))
def getMeanData(df,h,w):
    filterd_df = (df [(df ['h'] == h) & (df ['w'] == w)])
    mean = filterd_df.mean()
    min = filterd_df.min()
    max = filterd_df.max()
    return (min['time'],mean['time'],max['time']),(min['iterations'],mean['iterations'],max['iterations']) 

sizes = []
for h in range(2,10):
   for w in range(2,h+1):
     sizes.append((h,w,h*w))

sizes = sorted(sizes, key=lambda x: x[2])

print(sizes)

#headers = ['h','w','time','iters']

df_simple = pd.read_csv("DQN.csv",header=0)
df_memory = pd.read_csv("DQN_memory.csv",header=0)
df_target = pd.read_csv("DQN_target.csv",header=0)
df_full =   pd.read_csv("DQN_memory_target.csv",header=0)

dataFrames = [("simple",df_simple),("memory",df_memory),("target",df_target),("full",df_full)]

for df_name,df in dataFrames:
  print(df_name,getMeanData(df,6,6))
  print(df_name,getMeanData(df,9,4))

#for siple_sample in df_simple [ df_simple [''] ]

# Access data in the DataFrame using column names or indexing

