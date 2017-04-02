import pandas as pd
# import matplotlib.pyplot as plt
import statsmodels.api as sm
import elo as elo
import glicko as gl
import math
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from player_hits import *

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,}

nmatches = {"Afghanistan":0, "Australia":0,"Bangladesh":0,"England":0,"India":0,
"Ireland":0, "New Zealand":0,"Pakistan":0,"South Africa":0,"Sri Lanka":0,
"West Indies":0,"Zimbabwe":0,}

ratings_id = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
"Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
"West Indies":1000,"Zimbabwe":1000,}

elo_params = {"k_factor":10, "rating_class":float, "initial":1200, "beta":200,
    "margin_run":0.2, "margin_run_norm":50., "margin_wkts":0.2,
    "k_factor_run":10, "k_factor_wkts":10}

# elo_params_h = {"rating_class":float, "initial":1200, "beta":200,
#     "kf_wt_rating":.1, "kf_wt_margin_runs":.1, "kf_wt_margin_wkts":.1,
#     "kf_wt_winnerby":.1,"kf_wt_tossdecision":.1,"kf_wt_tosswinner":.1,
#     "kf_wt_bats": 1, "kf_wt_bowls": 1}

# elo_params_h = {'kf_wt_margin_runs': 0.09821003919240738, 'kf_wt_tosswinner': 0.09209440469651914, 'kf_wt_bowls': 2.725574877120683, 'initial': 800.0, 'kf_wt_bats': 3.6226754841144877, 'kf_wt_rating': 0.09983311288743675, 'beta': 50.0, 'kf_wt_margin_wkts': 0.06099484255128075, 'kf_wt_tossdecision': 0.09738668676738575, 'kf_wt_winnerby': 0.028766334691990383}
elo_params_h = {'kf_wt_bats': 0.9267052648247908, 'kf_wt_rating': 0.09042325925423486, 'beta': 50.0, 'initial': 1000.0, 'kf_wt_bowls': 0.37889936079352615}

glicko_params = {"mu":1500, "phi": 350, "sigma":0.06, "tau":1.0, "epsilon":0.000001,
    "Q":math.log(10)/400}

visual = False
bmen = pickle.load(open('../data/batsmen.pkl'))
bler = pickle.load(open('../data/bowlers.pkl'))

def get_bowl_diff(h, tlist, winner_id, loser_id):
    if type(h) is list:
        return 0
    wlist = [np.where(bler==i)[0][0] for i in tlist[winner_id] if i in bler]
    llist = [np.where(bler==i)[0][0] for i in tlist[loser_id] if i in bler]    
    return np.sum(np.asarray(sorted(map(h.get, wlist), reverse=True)[:5])-np.asarray(sorted(map(h.get, llist), reverse=True)[:5]))

def get_bat_diff(a, tlist, winner_id, loser_id):
    if type(a) is list:
        return 0
    wlist = [len(bler)+np.where(bmen==i)[0][0] for i in tlist[winner_id] if i in bmen]
    llist = [len(bler)+np.where(bmen==i)[0][0] for i in tlist[loser_id] if i in bmen]    
    return np.sum(np.asarray(sorted(map(a.get, wlist), reverse=True)[:6])-np.asarray(sorted(map(a.get, llist), reverse=True)[:6]))

def get_ratings(eloparams = None, hitsparams = None):
    env = elo.HitsElo(**eloparams)
    ratings = []
    df_train = pd.read_csv("../data/cricket.csv")

    # hlist = pickle.load(open('../data/player_hlist.pkl'))
    # alist = pickle.load(open('../data/player_alist.pkl'))

    hlist, alist = get_hubs_auth(hitsparams)

    tlist = pickle.load(open('../data/player_teams.pkl'))
    df_train.sort(columns="Date", inplace=True)
    ## Additional preprocessing for feats
    df_train["Toss_Winner"] = df_train["Toss_Winner"]==df_train["Team1"]
    df_train["Toss_Decision"] = LabelEncoder().fit_transform(df_train["Toss_Decision"])
    df_train["Winnerby"] = LabelEncoder().fit(["runs","wickets"]).transform(df_train["Winnerby"])

    for i in ratings_id.keys():
        ratings_id[i] = env.create_rating(value=1000)

    for i in range(len(df_train.index)):
        if df_train.Team1[i] not in teams_id.keys():
            ratings.append(ratings_id.copy()) #changed for evaluation
            continue
        if df_train.Team2[i] not in teams_id.keys():
            ratings.append(ratings_id.copy())
            continue    

        nmatches[df_train.Team1[i]] += 1
        nmatches[df_train.Team2[i]] += 1
        loser_rate = ratings_id[df_train.Team1[i]]
        winner_rate = ratings_id[df_train.Winner[i]]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_rate = ratings_id[df_train.Team2[i]]

        loser_id = df_train.Team1[i]
        winner_id = df_train.Winner[i]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_id = df_train.Team2[i]

        feats = {}
        feats["kf_wt_margin"] = df_train.Margin[i]
        feats["kf_wt_winnerby"] = df_train.Winnerby[i]
        feats["kf_wt_tossdecision"] = df_train.Toss_Decision[i]
        feats["kf_wt_tosswinner"] = df_train.Toss_Winner[i]
        feats["kf_wt_bats"] = get_bat_diff(alist[i], tlist, winner_id, loser_id)
        feats["kf_wt_bowls"] = get_bowl_diff(hlist[i], tlist, winner_id, loser_id)
        ratings_id[winner_id], ratings_id[loser_id] = env.rate_1vs1(winner_rate, loser_rate, 
            feats, exp=True)
        # print loser_id, winner_id, rate_1vs1(winner_rate, loser_rate)
        # print ratings_id
        ratings.append(ratings_id.copy())
    return ratings

def visualize(ratings):
    full_ratings = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
    "Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
    "West Indies":1000,"Zimbabwe":1000,}
    xr = range(len(ratings))

    # for i in range(len(ratings)):
    #     mxv = max(ratings[i].values())
    #     for k,v in ratings[i].items():
    #         ratings[i][k] = v/mxv

    tsum = sum(ratings[0].values())
    for i in range(len(ratings)):
        mxv = (tsum-sum(ratings[i].values()))/12.
        for k,v in ratings[i].items():
            ratings[i][k] = v+mxv

    for i in full_ratings.keys():
        full_ratings[i] = [j[i] for j in ratings]
        lowess = sm.nonparametric.lowess(full_ratings[i], xr, frac=0.2)
        plt.plot(lowess[:, 0], lowess[:, 1], label=i)
        plt.legend()

    plt.xlabel("Matches")
    plt.ylabel("Ratings")
    plt.legend(prop={'size':10}, ncol = 3)
    plt.show()

def vis_nmatches():
    plt.bar(range(len(nmatches)), nmatches.values(), align='center')
    plt.xticks(range(len(nmatches)), nmatches.keys())
    plt.show()

def rolling_validate(ratings, starti = 0.50, endi = 1, beta=80):
    df_train = pd.read_csv("../data/cricket.csv")
    df_train.sort(columns="Date", inplace=True)

    env = elo.Elo(beta=beta)
    start = int(len(df_train.index)*starti)
    end = int(len(df_train.index)*endi)
    err = 0
    err_log = 0
    for i in range(start, end):
        if df_train.Team1[i] not in teams_id.keys():
            continue
        if df_train.Team2[i] not in teams_id.keys():
            continue

        if ratings[i][df_train.Team1[i]] > ratings[i][df_train.Team2[i]]:
            if df_train.Winner[i] == df_train.Team2[i]:
                err += 1
        if ratings[i][df_train.Team1[i]] < ratings[i][df_train.Team2[i]]:
            if df_train.Winner[i] == df_train.Team1[i]:
                err += 1

        ## Function for log-loss
        pa = env.expect(ratings[i][df_train.Team1[i]], ratings[i][df_train.Team2[i]])
        pb = 1-pa
        if df_train.Winner[i] == df_train.Team1[i]:
            err_log += -math.log(pa)*0.5
        else:
            err_log += -math.log(pb)*0.5

    print "Accuracy: ", 1 - 1.*err/(end-start)
    print "Log Loss: ", err_log/(end-start)
    return 1.*err/(end-start)

# ratings = get_ratings(params=elo_params_h)
# if visual:
#     visualize(ratings)

# vis_nmatches()
# 1-rolling_validate(ratings, starti = 0.75, endi = 1, beta=elo_params_h["beta"])