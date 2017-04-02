import pickle
import numpy as np
import igraph as ig
import igraph.drawing
import networkx as nx
import matplotlib.pyplot as plt

def load_pickle():
    fp = open('../data/player_data.pkl')
    plist = pickle.load(fp)
    bats = np.unique([i[0] for i in plist[-1].keys()])
    bowls = np.unique([i[1] for i in plist[-1].keys()])
    return plist, bats, bowls

def get_hubs_auth():
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

def player_graph():
    plist, bats, bowls = load_pickle()
    plist = plist[-1]
    pl = np.unique(np.append(bats, bowls))
    msize = len(pl)
    playermat = [[0 for i in range(msize)] for j in range(msize)]
    for key, value in plist.iteritems():
        playermat[np.where(pl==key[1])[0][0]][np.where(pl==key[0])[0][0]] = value/100.
    
    G = nx.from_numpy_matrix(np.array(playermat))
    h, a = nx.hits(G)
    g = G.copy()
    for n in g.nodes():
        if (h[n] < 0.0050) or (a[n] < 0.0050):
            g.remove_node(n)
#    nx.draw(g)
    G = g.copy()
    h, a = nx.hits(G)
    playfin = nx.to_numpy_matrix(G)
#    pl = 20
#    playfin = [[0 for i in range(2*pl)] for j in range(2*pl)]
#    for i in range(pl):
#        for j in range(pl):
#            print i, j, h[i], a[j]
#            playfin[i][len(h)+j] = playermat[h[i]][a[j]]
    
    gi = ig.Graph.Full(n = len(h), directed = True)
    gi = gi.Weighted_Adjacency(playfin.tolist(), mode = "DIRECTED")
    gi.vs["name"] = pl[h.keys()].tolist()
    #g.delete_vertices(np.where(np.array(g.vs.degree())==0)[0])
    save_graph(gi, pl[h.keys()].tolist(), h.values())

def save_graph(g, vl, cl, layoutname = "fruchterman_reingold"):
    if layoutname == "kk":
        g.write_svg("newgraph.svg", labels = "name" , layout = g.layout(layoutname))
    else:
        igraph.plot(g, 'newgraph.png', layout=g.layout(layoutname), bbox=(500, 500), margin= 50, hovermode='closest', edge_width = g.es["weight"], 
            vertex_size=20, vertex_label=vl, vertex_color = 'green', vertex_label_dist = 2, vertex_label_size = 10, edge_arrow_size=0.5)


player_graph()
#plt.bar(range(len(h)), h.values(), align='center')
#plt.xticks(range(len(h)), h.keys())
#plt.show(h)


#plist, bats, bowls = load_pickle()
#plist = plist[-1]
#
##%%
#pl = np.unique(np.append(bats, bowls))
#msize = len(pl)
#playermat = [[0 for i in range(msize)] for j in range(msize)]
#for key, value in plist.iteritems():
#    playermat[np.where(pl==key[1])[0][0]][np.where(pl==key[0])[0][0]] = value/100.
#
#G = nx.from_numpy_matrix(np.array(playermat))
#h, a = nx.hits(G)
#
##%%
##h = sorted(h, key=h.get, reverse=True)
##a = sorted(a, key=a.get, reverse=True)
#g = G.copy()
#for n in g.nodes():
#    if (h[n] < 0.0050) or (a[n] < 0.0050):
#        g.remove_node(n)
#nx.draw(g)
#
##%%
#
#G = g.copy()
#h, a = nx.hits(G)
#playfin = nx.to_numpy_matrix(G)
##%%
#pl = len(h)
#playfin = [[0 for i in range(pl)] for j in range(pl)]
#for i in range(pl):
#    for j in range(pl):
#        print i, j, h[i], a[j]
#        playfin[i][j] = playermat[h[i]][a[j]]
#
##%%
#gi = ig.Graph.Full(n = len(h), directed = True)
#gi = gi.Weighted_Adjacency(playfin.tolist(), mode = "DIRECTED")
#gi.vs["name"] = pl[h.keys()].tolist()
##g.delete_vertices(np.where(np.array(g.vs.degree())==0)[0])
#save_graph(gi, pl[h.keys()].tolist(), h.values())