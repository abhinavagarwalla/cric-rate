import pandas as pd
from numpy import *
import numpy as np
import igraph as ig
import igraph.drawing
from sklearn.metrics import log_loss
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

visual = True

def visualize(ratings):
    full_ratings = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
    "Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
    "West Indies":1000,"Zimbabwe":1000,}
    xr = range(len(ratings))
    for i in range(len(full_ratings.keys())):
        full_ratings[full_ratings.keys()[i]] = [j[i] for j in ratings]
        lowess = sm.nonparametric.lowess(full_ratings[full_ratings.keys()[i]], xr, frac=0.2)
        plt.plot(lowess[:, 0], lowess[:, 1], label=full_ratings.keys()[i])
        plt.legend()
    print full_ratings.keys()
    plt.xlabel("Matches")
    plt.ylabel("Ratings")
    plt.legend(loc = 9,prop={'size':10}, ncol = 3)
    plt.show()

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
    return 1.*err/(end-start)
    # return log_loss(y_true, y_pred)

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,
}

def save_graph(g, layoutname = "fruchterman_reingold"):
    if layoutname == "kk":
        g.write_svg("newgraph.svg", labels = "name" , layout = g.layout(layoutname))
    else:
        igraph.plot(g, 'newgraph.png', layout=g.layout(layoutname), bbox=(1000, 1000), margin= 100, hovermode='closest', vertex_label = g.vs["name"], edge_width = g.es["weight"], vertex_color = "green", vertex_label_dist = 2, vertex_label_size = 25)
 
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
    elif method_no == 5:
        for l in range(len(teams_id)):
            for m in range(len(teams_id)):
                if played[l][m]!=0:
                    a = winmatrix[l][m] / (1.*played[l][m])
                    b = runmatrix[l][m] / (1.*played[l][m])
                    c = movrmatrix[l][m] / (1.*played[l][m])
                    d = (movrmatrix[l][m] / (1.*played[l][m]))* (winrmatrix[l][m] / (1.*(played[l][m])))
                    weight_matrix[l][m] =  0.25*(a + b + c + d)
    return weight_matrix

linkmatrix = [[] for i in range(len(teams_id))]

winmatrix, winrmatrix, winwmatrix, movrmatrix, movwmatrix, runmatrix, weight_matrix, played = ([[0 for j in range(len(teams_id))] for i in range(len(teams_id))] for no_of_matrix in range(8))

df_train = pd.read_csv("../data/cricket.csv")
df_train.sort(columns="Date", inplace=True)

pr1, pr2, pr3, pr4, pr5 = ([[0 for j in range(len(teams_id))] for i in range(len(df_train.index))] for loop in range(5))

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

    # g = ig.Graph.Full(n = len(teams_id), directed = True)
    # g = g.Weighted_Adjacency(create_weight_matrix(1), mode = "DIRECTED")
    # # g = g.Weighted_Adjacency(winmatrix, mode = "DIRECTED")
    # g.vs["name"] = teams_id.keys()
    # # print g.es["weight"]
    # pr1[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85))
    # save_graph(g)

#******************* Heatmap ******************
# a = np.array(winmatrix)
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(a)
# cbar = plt.colorbar(heatmap)
# cbar.set_label('No on matches won by Team1 against Team2', rotation=270, labelpad = +20)
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()
# column_labels = teams_id.keys()
# row_labels = teams_id.keys()
# ax.set_xticklabels(column_labels, minor=False, rotation=45)
# ax.set_yticklabels(row_labels, minor=False)
# ax.set_xticks(np.arange(a.shape[1]) + 0.5, minor=False)
# ax.set_yticks(np.arange(a.shape[0]) + 0.5, minor=False)
# plt.xlabel('Team2')
# plt.ylabel('Team1')
# plt.show()
#************************************************

    # g = ig.Graph.Full(n = len(teams_id), directed = True)
    # g = g.Weighted_Adjacency(create_weight_matrix(2), mode = "DIRECTED")
    # g.vs["name"] = teams_id.keys()
    # pr2[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85))

    # g = ig.Graph.Full(n = len(teams_id), directed = True)
    # g = g.Weighted_Adjacency(create_weight_matrix(3), mode = "DIRECTED")
    # g.vs["name"] = teams_id.keys()
    # pr3[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85))

    # g = ig.Graph.Full(n = len(teams_id), directed = True)
    # g = g.Weighted_Adjacency(create_weight_matrix(4), mode = "DIRECTED")
    # g.vs["name"] = teams_id.keys()
    # pr4[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85))

    # g = ig.Graph.Full(n = len(teams_id), directed = True)
    # g = g.Weighted_Adjacency(create_weight_matrix(4), mode = "DIRECTED")
    # g.vs["name"] = teams_id.keys()
    # pr4[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85))

    # g = ig.Graph.Full(n = len(teams_id), directed = True)
    # g = g.Weighted_Adjacency(create_weight_matrix(5), mode = "DIRECTED")
    # g.vs["name"] = teams_id.keys()
    # pr5[i] = (g.pagerank(vertices = None, directed = True, damping = 0.85))
    # print weight_matrix
    # print pr2

# visualize(pr5)
# print "Accuracy for Weighting function 1 : ", (1 - rolling_validate(pr1, start_index, end_index))
# print "Accuracy for Weighting function 2 :", (1 - rolling_validate(pr2, start_index, end_index))
# print "Accuracy for Weighting function 3 :", (1 - rolling_validate(pr3, start_index, end_index))
# print "Accuracy for Weighting function 4 :", (1 - rolling_validate(pr4, start_index, end_index))
# print "Accuracy for Weighting function 5 :", (1 - rolling_validate(pr5, start_index, end_index))