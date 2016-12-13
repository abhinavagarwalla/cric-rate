from trueskill import TrueSkill, Rating, quality_1vs1, rate_1vs1
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

teams_id = {"Afghanistan":0, "Australia":1,"Bangladesh":2,"England":3,"India":4,
"Ireland":5, "New Zealand":6,"Pakistan":7,"South Africa":8,"Sri Lanka":9,
"West Indies":10,"Zimbabwe":11,}

ratings_id = {"Afghanistan":1000, "Australia":1000,"Bangladesh":1000,"England":1000,"India":1000,
"Ireland":1000, "New Zealand":1000,"Pakistan":1000,"South Africa":1000,"Sri Lanka":1000,
"West Indies":1000,"Zimbabwe":1000,}

def main():
    env = TrueSkill()
    ratings = []
    df_train = pd.read_csv("../data/cricket.csv")
    df_train.sort(columns="Date", inplace=True)

    for i in ratings_id.keys():
        ratings_id[i] = env.create_rating()

    for i in df_train.index:
        if df_train.Team1[i] not in teams_id.keys():
            continue
        if df_train.Team2[i] not in teams_id.keys():
            continue    

        loser_rate = ratings_id[df_train.Team1[i]]
        winner_rate = ratings_id[df_train.Winner[i]]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_rate = ratings_id[df_train.Team2[i]]

        loser_id = df_train.Team1[i]
        winner_id = df_train.Winner[i]
        if df_train.Team1[i] in df_train.Winner[i]:
            loser_id = df_train.Team2[i]

        ratings_id[winner_id], ratings_id[loser_id] = rate_1vs1(winner_rate, loser_rate)
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

ratings = main()
visualize(ratings)
