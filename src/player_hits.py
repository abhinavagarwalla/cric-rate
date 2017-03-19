import pickle
import numpy as np
import igraph as ig
import igraph.drawing
import networkx as nx
import matplotlib.pyplot as plt

def load_pickle():
    fp = open('../data/player_data.pkl')
    plist = pickle.load(fp)
#    print len(plist[-1])
#    for i in plist:
#       print len(i.keys())
    bats = np.unique([i[0] for i in plist[-1].keys()])
    bowls = np.unique([i[1] for i in plist[-1].keys()])
    return plist, bats, bowls


plist, bats, bowls = load_pickle()
hlist, alist = [], []

for pl in range(len(plist)):
    print pl
    try:
        playermat = np.zeros((len(bowls)+len(bats), len(bowls)+len(bats)))
        for key, value in plist[pl].iteritems():
            playermat[np.where(bowls==key[1])[0][0]][len(bowls)+np.where(bats==key[0])[0][0]] = value
        G = nx.from_numpy_matrix(playermat)
        h, a = nx.hits(G)
        hlist.append(h)
        alist.append(a)
    except:
        hlist.append([0])
        alist.append([0])

with open('../data/player_hlist.pkl', 'w') as fp:
    pickle.dump(hlist, fp)

with open('../data/player_alist.pkl', 'w') as fp:
    pickle.dump(alist, fp)

with open('../data/batsmen.pkl', 'w') as fp:
    pickle.dump(bats, fp)

with open('../data/bowlers.pkl', 'w') as fp:
    pickle.dump(bowls, fp)
#plt.bar(range(len(h)), h.values(), align='center')
#plt.xticks(range(len(h)), h.keys())
#plt.show(h)