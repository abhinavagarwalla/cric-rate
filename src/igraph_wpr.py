import pandas as pd
from numpy import *
import numpy as np
import igraph as ig
import igraph.drawing

def rolling_validate(ratings, starti, endi):
    df_train = pd.read_csv("../data/cricket.csv")
    df_train.sort(columns="Date", inplace=True)

    start = int(len(df_train.index)*starti)
    end = int(len(df_train.index)*endi)
    err = 0
    for i in range(start, end):
        if df_train.Team1[i] not in teams_id.keys():
            continue
        if df_train.Team2[i] not in teams_id.keys():
            continue

        if ratings[i][teams_id[df_train.Team1[i]]] > ratings[i][teams_id[df_train.Team2[i]]]:
            if df_train.Winner[i] == df_train.Team2[i]:
                err += 1
        if ratings[i][teams_id[df_train.Team1[i]]] < ratings[i][teams_id[df_train.Team2[i]]]:
            if df_train.Winner[i] == df_train.Team1[i]:
                err += 1
    print err
    return 1.*err/(end-start)

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,
}

linkmatrix = [[] for i in range(len(teams_id))]
winmatrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
winrmatrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
winwmatrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
movrmatrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
movwmatrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
runmatrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]
weight_matrix = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]

played = [[0 for j in range(len(teams_id))] for i in range(len(teams_id))]


df_train = pd.read_csv("../data/cricket.csv")
df_train.sort(columns="Date", inplace=True)

pr = [[0 for j in range(len(teams_id))] for i in range(len(df_train.index))]

start_index = 0.75
end_index = 1.0

print len(df_train.index)*end_index

for i in range(len(df_train.index)):
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
    # if "runs" in df_train.Winnerby[i]:
    #     winrmatrix[loser_id][winner_id] += 1
    #     movrmatrix[loser_id][winner_id] += df_train.Margin[i]
    # elif "wickets" in df_train.Winnerby[i]:
    #     winwmatrix[loser_id][winner_id] += 1
    #     movrmatrix[loser_id][winner_id] += df_train.Margin[i]
    # toss_winner_id = teams_id[df_train.Toss_Winner[i]]
    # toss_loser_id = teams_id[df_train.Team1[i]]
    # if df_train.Team1[i] in df_train.Toss_Winner[i]:
    #     toss_loser_id = teams_id[df_train.Team2[i]]
    # runmatrix[toss_loser_id][toss_winner_id] += df_train.Run1[i]
    # runmatrix[toss_winner_id][toss_loser_id] += df_train.Run2[i]

    g = ig.Graph.Full(n = len(teams_id), directed = True)
    g = g.Adjacency(winmatrix, mode = "DIRECTED")
    g.vs["name"] = teams_id.keys()

    layout = g.layout("fruchterman_reingold")
    # igraph.plot(g, '2.png', layout=layout, bbox=(1000, 1000), margin=120, hovermode='closest', labels = "name")

    # layout = g.layout("kk")
    # g.write_svg("newgraph1.svg", labels = "name" , layout=layout)

    for l in range(len(teams_id)):
        for m in range(len(teams_id)):
            if played[l][m]!=0:
                weight_matrix[l][m] = winmatrix[l][m] / (1.*(played[l][m]))

    # print weight_matrix
    pr[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85, weights = weight_matrix, implementation="power"))

print len(pr)

print (1 - rolling_validate(pr, start_index, end_index))

exit()