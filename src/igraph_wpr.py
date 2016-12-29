import pandas as pd
from numpy import *
import numpy as np
import igraph as ig
import igraph.drawing
from sklearn.metrics import log_loss
import math

def rolling_validate(ratings, starti, endi):
    df_train = pd.read_csv("../data/cricket.csv")
    df_train.sort(columns="Date", inplace=True)

    start = int(len(df_train.index)*starti)
    end = int(len(df_train.index)*endi)
    err = 0
    y_true = []
    y_pred = []
    for i in range(start, end):
        if df_train.Team1[i] not in teams_id.keys():
            continue
        if df_train.Team2[i] not in teams_id.keys():
            continue

        team_probs = np.zeros(len(teams_id))
        rating1 = ratings[i][teams_id[df_train.Team1[i]]]
        rating2 = ratings[i][teams_id[df_train.Team2[i]]]
        team_probs[teams_id[df_train.Team1[i]]] = 1./(1+math.exp(rating2-rating1))
        team_probs[teams_id[df_train.Team2[i]]] = 1./(1+math.exp(rating1-rating2))
        y_pred.append(team_probs)
        y_true.append(df_train.Winner[i])
        if rating1 > rating2:
            if df_train.Winner[i] == df_train.Team2[i]:
                err += 1            
        if rating1 < rating2:
            if df_train.Winner[i] == df_train.Team1[i]:
                err += 1
    # return 1.*err/(end-start)
    return log_loss(y_true, y_pred)

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,
}

def save_graph(g, layoutname = "fruchterman_reingold"):
    if layoutname == "kk":
        g.write_svg("newgraph.svg", labels = "name" , layout = g.layout(layoutname))
    else:
        igraph.plot(g, 'newgraph.png', layout=g.layout(layoutname), bbox=(1000, 1000), margin=120, hovermode='closest', labels = "name")

def create_weight_matrix(method_no):
    if method_no == 1:
        # percentage of matches won 
        for l in range(len(teams_id)):
            for m in range(len(teams_id)):
                if played[l][m]!=0:
                    weight_matrix[l][m] = winmatrix[l][m] / (1.*(played[l][m]))
    elif method_no == 2:
        # 
        for l in range(len(teams_id)):
            for m in range(len(teams_id)):
                if played[l][m]!=0:
                    weight_matrix[l][m] = runmatrix[l][m] / (1.*(played[l][m]))
    elif method_no == 3:
        # 
        for l in range(len(teams_id)):
            for m in range(len(teams_id)):
                if played[l][m]!=0:
                    weight_matrix[l][m] = movrmatrix[l][m] / (1.*(played[l][m]))
    elif method_no == 4:
        # 
        for l in range(len(teams_id)):
            for m in range(len(teams_id)):
                if played[l][m]!=0:
                    weight_matrix[l][m] = ((movrmatrix[l][m] / (1.*(played[l][m])))* (winrmatrix[l][m] / (1.*(played[l][m]))))
    return weight_matrix

linkmatrix = [[] for i in range(len(teams_id))]

winmatrix, winrmatrix, winwmatrix, movrmatrix, movwmatrix, runmatrix, weight_matrix, played = ([[0 for j in range(len(teams_id))] for i in range(len(teams_id))] for no_of_matrix in range(8))

df_train = pd.read_csv("../data/cricket.csv")
df_train.sort(columns="Date", inplace=True)

pr1, pr2, pr3, pr4 = ([[0 for j in range(len(teams_id))] for i in range(len(df_train.index))] for loop in range(4))

start_index = 0.75
end_index = 1.0

# print len(df_train.index)*end_index

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
    if "runs" in df_train.Winnerby[i]:
        winrmatrix[loser_id][winner_id] += 1
        movrmatrix[loser_id][winner_id] += df_train.Margin[i]
    elif "wickets" in df_train.Winnerby[i]:
        winwmatrix[loser_id][winner_id] += 1
        movrmatrix[loser_id][winner_id] += df_train.Margin[i]
    toss_winner_id = teams_id[df_train.Toss_Winner[i]]
    toss_loser_id = teams_id[df_train.Team1[i]]
    if df_train.Team1[i] in df_train.Toss_Winner[i]:
        toss_loser_id = teams_id[df_train.Team2[i]]
    runmatrix[toss_loser_id][toss_winner_id] += df_train.Run1[i]
    runmatrix[toss_winner_id][toss_loser_id] += df_train.Run2[i]

    g = ig.Graph.Full(n = len(teams_id), directed = True)
    g = g.Adjacency(winmatrix, mode = "DIRECTED")
    g.vs["name"] = teams_id.keys()

    # save_graph(g)

    pr1[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85,
        weights = create_weight_matrix(1), implementation="power"))
    pr2[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85,
        weights = create_weight_matrix(2), implementation="power"))
    pr3[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85,
        weights = create_weight_matrix(3), implementation="power"))
    pr4[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85,
        weights = create_weight_matrix(4), implementation="power"))
    
    # print weight_matrix
print np.sum(np.array(pr1)-np.array(pr2))
print pr1[:10], pr2[:10]
print "Accuracy for Weighting function 1 : ", (1 - rolling_validate(pr1, start_index, end_index))
print "Accuracy for Weighting function 2 :", (1 - rolling_validate(pr2, start_index, end_index))
print "Accuracy for Weighting function 3 :", (1 - rolling_validate(pr3, start_index, end_index))
print "Accuracy for Weighting function 4 :", (1 - rolling_validate(pr4, start_index, end_index))