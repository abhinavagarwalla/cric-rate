import pandas as pd
from numpy import *
import numpy as np
import igraph as ig

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,
}
linkmatrix = [[] for i in range(len(teams_id))]
winmatrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
played = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
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
    played[loser_id][winner_id] += 1
    played[winner_id][loser_id] += 1
    winmatrix[loser_id][winner_id] += 1

print winmatrix

g = ig.Graph.Full(n = len(teams_id), directed = True)
g = g.Weighted_Adjacency(winmatrix, mode = "DIRECTED")
g.vs["name"] = teams_id.keys()
layout = g.layout("kk")
# g.write_svg("graph.svg", labels = "name" , layout=layout)
print g.ecount()
# print g

for i in range(len(teams_id)):
    for j in range(len(teams_id)):
        if played[i][j]!=0:
            winmatrix[i][j] /= 1.*(played[i][j])

pr = g.personalized_pagerank(vertices = None, directed = True, damping = 0.85,
    weights = winmatrix, implementation='power')
print pr
exit()