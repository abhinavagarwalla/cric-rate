import pandas as pd
from numpy import *
import numpy as np
import igraph as ig

def pageRankGenerator(At = [array((), int64)], 
                      numLinks = array((), int64),  
                      ln = array((), int64),
                      alpha = 0.85, 
                      convergence = 0.0001, 
                      checkSteps = 10,
                      probweights = [array((), int64)],
                      outwardstrength = array((), int64)
                      ):
    N = len(At)

    M = ln.shape[0]

    iNew = ones((N,), float64) / N
    iOld = ones((N,), float64) / N

    """
    probweights = [[0,1,0.666666666666667,0.529411764705882,0.545454545454545,0.733333333333333,0.384615384615385,0.5,0.85,1],
                    [0,0,0.285714285714286,0.2,0.615384615384615,0.0714285714285714,0,0.142857142857143,0.583333333333333,0.615384615384615],
                    [0.333333333333333,0.714285714285714,0,0.285714285714286,0.416666666666667,0.777777777777778,0.714285714285714,0.384615384615385,0.769230769230769,1],
                    [0.470588235294118,0.8,0.714285714285714,0,0.647058823529412,0.461538461538462,0.416666666666667,0.636363636363636,0.619047619047619,0.714285714285714],
                    [0.454545454545455,0.384615384615385,0.583333333333333,0.352941176470588,0,0.6,0.375,0.454545454545455,0.416666666666667,0.857142857142857],
                    [0.266666666666667,0.928571428571429,0.222222222222222,0.538461538461538,0.4,0,0.368421052631579,0.428571428571429,0.6875,0.857142857142857],
                    [0.615384615384615,1,0.285714285714286,0.583333333333333,0.625,0.631578947368421,0,0.357142857142857,1,1],
                    [0.5,0.857142857142857,0.615384615384615,0.363636363636364,0.545454545454545,0.571428571428571,0.642857142857143,0,0.333333333333333,0.9],
                    [0.15,0.416666666666667,0.230769230769231,0.380952380952381,0.583333333333333,0.3125,0,0.666666666666667,0,0.875],
                    [0,0.384615384615385,0,0.285714285714286,0.142857142857143,0.142857142857143,0,0.1,0.125,0]]

    outwardstrength = [2.79,6.49,3.60,3.52,4.52,4.30,2.90,3.67,5.38,7.82]
    """
    
    finalweights = probweights
    for i in range(len(probweights)):
        for j in range(len(probweights[i])):
            finalweights[i][j] /= outwardstrength[j]

    print finalweights
            
    done = False
    while not done:

        iNew /= sum(iNew) #normalization for convergence

        for step in range(checkSteps):

            iOld, iNew = iNew, iOld

            #print sum(iOld)
            #raw_input("Enter anythin")
            oneIv = (1 - alpha) * sum(iOld) / N

            oneAv = 0.0
            if M > 0:
                oneAv = alpha * sum(iOld.take(ln, axis = 0)) / N

            ii = 0 
            while ii < N:
                page = At[ii]
                h = 0
                if page.shape[0]:
                    h = alpha * dot(
                            iOld.take(page, axis = 0),
                            1. / array(finalweights[ii]).take(page, axis = 0)
                            )
                #print array(finalweights[ii]).take(page, axis = 0)
                #print iOld
                #print iOld.take(page, axis = 0)
               # raw_input("Enter ")
                iNew[ii] = h + oneAv + oneIv
                ii += 1

        diff = sum(abs(iNew - iOld))
        done = (diff < convergence)

        yield iNew


def transposeLinkMatrix(
        outGoingLinks = [[]]
        ):
 
    nPages = len(outGoingLinks)
    incomingLinks = [[] for ii in range(nPages)]
    numLinks = zeros(nPages, int64)
    leafNodes = []

    for ii in range(nPages):
        if len(outGoingLinks[ii]) == 0:
            leafNodes.append(ii)
        else:
            numLinks[ii] = len(outGoingLinks[ii])
            for jj in outGoingLinks[ii]:
                incomingLinks[jj].append(ii)

    incomingLinks = [array(ii) for ii in incomingLinks]
    numLinks = array(numLinks)
    leafNodes = array(leafNodes)

    return incomingLinks, numLinks, leafNodes


def pageRank(
        linkMatrix = [[]],
        alpha = 0.85, 
        convergence = 0.0001, 
        checkSteps = 10,
        probweights = [array((), int64)],
        outwardstrength = array((), int64)
        ):

    incomingLinks, numLinks, leafNodes = transposeLinkMatrix(linkMatrix)

    for gr in pageRankGenerator(incomingLinks, numLinks, leafNodes,
                                alpha = alpha, convergence = convergence,
                                checkSteps = checkSteps, probweights=probweights, 
                                outwardstrength=outwardstrength):
        final = gr

    return final


def main():

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

    for i in range(len(teams_id)):
        for j in range(len(teams_id)):
            if played[i][j]!=0:
                winmatrix[i][j] /= 1.*(played[i][j])
     
    # print winmatrix
    
    wnm = np.array(winmatrix)
    outs = np.sum(wnm, axis=1)
    pr =  pageRank(linkmatrix, alpha=0.85, convergence=0.001, checkSteps=10, probweights=winmatrix, outwardstrength=outs)
    sum = 0
    for i in range(len(pr)):
        print i, "=", pr[i]
        sum = sum + pr[i]
    print "s = " + str(sum)

if __name__ == "__main__":
    main()
