import pickle
import numpy as np
import igraph as ig
import igraph.drawing
import networkx as nx
import matplotlib.pyplot as plt

fp = open('../data/player_data.pkl')
plist = pickle.load(fp)
print len(plist)
for i in plist:
	print len(i.keys())
exit()
bats = np.unique([i[0] for i in plist.keys()])
bowls = np.unique([i[1] for i in plist.keys()])

# playermat = [[0 for i in range(len(bowls)+len(bats))] for j in range(len(bowls)+len(bats))]
playermat = np.zeros((len(bowls)+len(bats), len(bowls)+len(bats)))
for key, value in plist.iteritems():
	playermat[np.where(bowls==key[1])[0][0]][len(bowls)+np.where(bats==key[0])[0][0]] = value

G = nx.from_numpy_matrix(playermat)
h, a = nx.hits(G)

plt.bar(range(len(h)), h.values(), align='center')
plt.xticks(range(len(h)), h.keys())
plt.show(h)