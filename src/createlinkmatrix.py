import pandas as pd
from pageRank import *
import numpy as np

'''
 Scotland, Kenya, Canada
'''

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,
}

linkmatrix = [[] for i in xrange(len(teams_id))]
df_train = pd.read_csv("../data/cricket.csv")

for i in df_train.index:
	if df_train.Team1[i] not in teams_id.keys():
		continue
	if df_train.Team2[i] not in teams_id.keys():
		continue	
	loser_id = teams_id[df_train.Team1[i]]
	winner_id = teams_id[df_train.Winner[i]]
	if df_train.Team1[i] in df_train.Winner[i]:
		loser_id = teams_id[df_train.Team2[i]]
	linkmatrix[loser_id].append(winner_id)


print linkmatrix
pr =  pageRank(linkmatrix, alpha=0.85, convergence=0.00001, checkSteps=1000)
sum = 0
for i in range(len(pr)):
    print i, "=", pr[i]
    sum = sum + pr[i]
print "s = " + str(sum)
