import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import elo as elo
import glicko as gl
from trueskill import TrueSkill, Rating, quality_1vs1, rate_1vs1
import math
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.preprocessing import LabelEncoder

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,}

ratings_id = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
"Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
"West Indies":1000,"Zimbabwe":1000,}

elo_params = {"k_factor":10, "rating_class":float, "initial":1200, "beta":200,
    "margin_run":0.2, "margin_run_norm":50., "margin_wkts":0.2,
    "k_factor_run":10, "k_factor_wkts":10}

elo_params_k = {"rating_class":float, "initial":1200, "beta":200,
    "kf_wt_rating":.1, "kf_wt_margin_runs":.1, "kf_wt_margin_wkts":.1,
    "kf_wt_winnerby":.1,"kf_wt_tossdecision":.1,"kf_wt_tosswinner":.1}

glicko_params = {"mu":1500, "phi": 350, "sigma":0.06, "tau":1.0, "epsilon":0.000001,
    "Q":math.log(10)/400}

visual = False

def get_ratings(method='elo', params=None):
    if method=='elo':
        env = elo.ModElo(**params)
    elif method=='trueskill':
        env = TrueSkill(**params)
    elif method=='glicko':    
        env = gl.ModGlicko2(**params)
    elif method=='elo_custom_k':
        env = elo.CustomElo(**params)
    else:
        print "Select a valid method"
        exit()

    ratings = []
    df_train = pd.read_csv("../data/cricket.csv")
    df_train.sort(columns="Date", inplace=True)
    ## Additional preprocessing for feats
    if method=='elo_custom_k':
        df_train["Toss_Winner"] = df_train["Toss_Winner"]==df_train["Team1"]
        df_train["Toss_Decision"] = LabelEncoder().fit_transform(df_train["Toss_Decision"])
        df_train["Winnerby"] = LabelEncoder().fit(["runs","wickets"]).transform(df_train["Winnerby"])

    for i in ratings_id.keys():
        ratings_id[i] = env.create_rating()

    for i in range(len(df_train.index)):
        if df_train.Team1[i] not in teams_id.keys():
            ratings.append(ratings_id.copy()) #changed for evaluation
            continue
        if df_train.Team2[i] not in teams_id.keys():
            ratings.append(ratings_id.copy())
            continue    

        loser_rate = ratings_id[df_train.Team1[i]]
        winner_rate = ratings_id[df_train.Winner[i]]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_rate = ratings_id[df_train.Team2[i]]

        loser_id = df_train.Team1[i]
        winner_id = df_train.Winner[i]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_id = df_train.Team2[i]

        if not method=='elo_custom_k':
            ratings_id[winner_id], ratings_id[loser_id] = env.rate_1vs1(winner_rate, loser_rate, 
                df_train.Winnerby[i], df_train.Margin[i])
        else:
            feats = {}
            feats["kf_wt_margin"] = df_train.Margin[i]
            feats["kf_wt_winnerby"] = df_train.Winnerby[i]
            feats["kf_wt_tossdecision"] = df_train.Toss_Decision[i]
            feats["kf_wt_tosswinner"] = df_train.Toss_Winner[i]
            ratings_id[winner_id], ratings_id[loser_id] = env.rate_1vs1(winner_rate, loser_rate, 
                feats)
        # print loser_id, winner_id, rate_1vs1(winner_rate, loser_rate)
        # print ratings_id
        ratings.append(ratings_id.copy())
    return ratings

def visualize(ratings):
    full_ratings = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
    "Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
    "West Indies":1000,"Zimbabwe":1000,}
    xr = range(len(ratings))
    for i in full_ratings.keys():
        full_ratings[i] = [j[i] for j in ratings]
        lowess = sm.nonparametric.lowess(full_ratings[i], xr, frac=0.2)
        plt.plot(lowess[:, 0], lowess[:, 1])
    plt.show()

def rolling_validate(ratings, starti = 0.50, endi = 1):
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

        if ratings[i][df_train.Team1[i]] > ratings[i][df_train.Team2[i]]:
            if df_train.Winner[i] == df_train.Team2[i]:
                err += 1
        if ratings[i][df_train.Team1[i]] < ratings[i][df_train.Team2[i]]:
            if df_train.Winner[i] == df_train.Team1[i]:
                err += 1

    return 1.*err/(end-start)

ratings = get_ratings(method = 'elo_custom_k', params=elo_params_k)
if visual:
    visualize(ratings)

print "Accuracy: ", 1-rolling_validate(ratings)