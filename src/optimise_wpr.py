import pandas as pd
from numpy import *
import numpy as np
import igraph as ig
import igraph.drawing
from sklearn.metrics import log_loss
import math
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def rolling_validate(ratings, starti, endi):
    df_train = pd.read_csv("../data/cricket.csv")
    df_train.sort(columns="Date", inplace=True)

    start = int(len(df_train.index)*starti)
    end = int(len(df_train.index)*endi)
    err = 0
    err_log = 0
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

    print "Accuracy: ", 1-1.*err/(end-start)
    print "Log Loss: ", log_loss(y_true, y_pred)
    return 1.*err/(end-start)

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,
}



df_train = pd.read_csv("../data/cricket.csv")
df_train.sort(columns="Date", inplace=True)

# print len(df_train.index)*end_index
def get_ratings(params):
    pr1 = [[0 for j in range(len(teams_id))] for i in range(len(df_train.index))]
    winmatrix, winrmatrix, winwmatrix, movrmatrix, movwmatrix, runmatrix, weight_matrix, played = ([[0 for j in range(len(teams_id))] for i in range(len(teams_id))] for no_of_matrix in range(8))
    linkmatrix = [[] for i in range(len(teams_id))]
    xa, xb, xc, xd = params["xa"], params["xb"], params["xc"], params["xd"]
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

        for l in range(len(teams_id)):
            for m in range(len(teams_id)):
                if played[l][m]!=0:
                    a = winmatrix[l][m] / (1.*played[l][m])
                    b = runmatrix[l][m] / (1.*played[l][m])
                    c = movrmatrix[l][m] / (1.*played[l][m])
                    d = (movrmatrix[l][m] / (1.*played[l][m]))* (winrmatrix[l][m] / (1.*(played[l][m])))
                    # print a,b,c,d
                    weight_matrix[l][m] = xa*a + xb*b + xc*c + xd*d

        g = ig.Graph.Full(n = len(teams_id), directed = True)
        g = g.Weighted_Adjacency(weight_matrix, mode = "DIRECTED")
        g.vs["name"] = teams_id.keys()
        pr1[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85))
    return pr1

start_index = 0.5
end_index = 0.75
max_evals = 100
def get_err(params):
    ratings = get_ratings(params)
    err = rolling_validate(ratings, start_index, end_index)
    print "  with params: ", params
    return {'loss': err, 'status': STATUS_OK}

wpr_params_grid = {"xa":hp.uniform("xa", -10.0, 10.0),
                    "xb":hp.uniform("xb", -10.0, 10.0),
                    "xc":hp.uniform("xc", -10.0, 10.0),
                    "xd":hp.uniform("xd", -10.0, 10.0)}
wpr_params = {"xa":0.0, "xb":1.0, "xc": 0., "xd": 0.}
best = fmin(get_err, wpr_params_grid, algo=tpe.suggest, trials=Trials(), max_evals=max_evals)

# print 1-rolling_validate(get_ratings(wpr_params), start_index, end_index)
print('\n\nBest Scoring Value')
print(best)

final_ratings = get_ratings(best)
print "Validation:  ", 1-rolling_validate(final_ratings, starti=0.5, endi=0.75)
print "Test: ", 1-rolling_validate(final_ratings, starti=0.75, endi=1)
