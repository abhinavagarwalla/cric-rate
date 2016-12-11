import pandas as pd
from pageRank import *

linkmatrix = [[] for i in xrange(10)]
'''
teams_id = {"Afghanistan":0,
"Australia":1,
"Bangladesh":2,
"Bermuda":3,
"Canada":4,
"England":5,
"HongKong":6,
"India":7,
"Ireland":8,
"Kenya":9,
"Netherlands":10,
"New Zealand":11,
"Pakistan":12,
"Scotland":13,
"South Africa":14,
"Sri Lanka":15,
"U.A.E.":16,
"West Indies":17,
"Zimbabwe":18,
}
'''

teams_id = {"Australia":0,"Bangladesh":1,"England":2,"India":3,
"New Zealand":4,"Pakistan":5,"South Africa":6,"Sri Lanka":7,
"West Indies":8,
"Zimbabwe":9,
}
xltrain = pd.ExcelFile("E:/kaggle/23yard/second-attempt/Train3.xlsx")
df_train = xltrain.parse("Sheet1")

for i in df_train.index:
    loser_id = teams_id[df_train.Loser[i]]
    winner_id = teams_id[df_train.Winner[i]]
    linkmatrix[loser_id].append(winner_id)

print linkmatrix
pr =  pageRank(linkmatrix, alpha=0.85, convergence=0.00001, checkSteps=1000)
sum = 0
for i in range(len(pr)):
    print i, "=", pr[i]
    sum = sum + pr[i]
print "s = " + str(sum)
