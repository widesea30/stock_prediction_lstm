import os
import pandas as pd

# Give the location of the excel file
loc = ("Output.xlsx")

# Number of epochs when training. Currently 3500 is enough, but as data is bigger, you'd better increase it.
nEpochs=3500

# batch_size when training. It is good that the value is less than 16.
batch_size=10

# the number of date considering history
nHistCount = 30

# the number of columns(model prediction result, last 7 days, last 14 days, last 21 days, last month
# 0 - use only prediction result of each model
# 1 - use prediction result, rights in last 7 days of each model
# 2 - use prediction result, rights in last 7&14 days of each model
# 3 - use prediction result, rights in last 7&14&21 days of each model
# 4 - use prediction result, rights in last 7&14&21&month days of each model
nColModel = 4

# use V-Y column or not
use_V = 0
use_W = 0
use_X = 0
use_Y = 0

loc_configfile = "config.csv"
if os.path.exists(loc_configfile):
    data = pd.read_csv(loc_configfile)
    nEpochs = data.values[0, 0]
    batch_size = data.values[0, 1]
    nHistCount = data.values[0, 2]
    nColModel = data.values[0, 3]
    use_V = data.values[0, 4]
    use_W = data.values[0, 5]
    use_X = data.values[0, 6]
    use_Y = data.values[0, 7]


