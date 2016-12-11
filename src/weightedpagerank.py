import pandas as pd
from numpy import *

def pageRankGenerator(At = [array((), int64)], 
                      numLinks = array((), int64),  
                      ln = array((), int64),
                      alpha = 0.85, 
                      convergence = 0.0001, 
                      checkSteps = 10
                      ):
    N = len(At)

    M = ln.shape[0]

    iNew = ones((N,), float64) / N
    iOld = ones((N,), float64) / N

    lostrunmatrix = [[741,151],[3024,217],[2069,143],[2701,147],
        [1607,141],[2363,141],[1482,73],[2086,201],[2756,171],[2459,201]]
    
    theta = [0.60,0.2,0.2]

    tindex = [3.05472137456351,2.41509628753863,2.62352812316561,3.00760778028791,2.79430464867665,2.89969477615909,3.02723019069511,
              3.19736240909449,2.58348410889911,2.35986927171653]

    
    
    matchrunwickets = [[numLinks[i],lostrunmatrix[i][0],lostrunmatrix[i][1]] for i in range(len(lostrunmatrix))]
    print matchrunwickets
    print theta

    weightloss = [dot(theta,matchrunwickets[i]) for i in range(len(matchrunwickets))]

    weightloss = array(weightloss)
    print weightloss.take(At[0],axis=0)
    print numLinks.take(At[0],axis=0)
    print numLinks, weightloss
    #return
    done = False
    while not done:

        iNew /= sum(iNew) #normalization for convergence

        for step in range(checkSteps):

            iOld, iNew = iNew, iOld

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
                            1. / weightloss.take(page, axis = 0)
                            )
                iNew[ii] = h + oneAv + oneIv* (tindex[ii]/sum(tindex))
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
        checkSteps = 10
        ):

    incomingLinks, numLinks, leafNodes = transposeLinkMatrix(linkMatrix)

    for gr in pageRankGenerator(incomingLinks, numLinks, leafNodes,
                                alpha = alpha, convergence = convergence,
                                checkSteps = checkSteps):
        final = gr

    return final


def main():
    linkmatrix = [[] for i in xrange(10)]

    
    teams_id = {"Australia":0,"Bangladesh":1,"England":2,"India":3,
    "New Zealand":4,"Pakistan":5,"South Africa":6,"Sri Lanka":7,
    "West Indies":8,
    "Zimbabwe":9,
    }
    xltrain = pd.ExcelFile("E:/kaggle/23yard/second-attempt/Train3.xlsx")
    df_train = xltrain.parse("Sheet1")

    for i in df_train.index:
        loser_id = teams_id[df_train.Loser[i]]
        winner_id = teams_id[df_train.Winner[i]]
        linkmatrix[loser_id].append(winner_id)

    #print linkmatrix
    pr =  pageRank(linkmatrix, alpha=0.85, convergence=0.00001, checkSteps=10)
    sum = 0
    for i in range(len(pr)):
        print i, "=", pr[i]
        sum = sum + pr[i]
    print "s = " + str(sum)

if __name__ == "__main__":
    main()
