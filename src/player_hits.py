import pickle
import numpy as np
# import igraph as ig
# import igraph.drawing
import networkx as nx
# import matplotlib.pyplot as plt

pickle_loaded = False

def load_pickle():
    print "Reading player_data.pkl file"
    fp = open('../data/player_data.pkl')
    plist = pickle.load(fp)
    bats = np.unique([i[0] for i in plist[-1].keys()])
    bowls = np.unique([i[1] for i in plist[-1].keys()])
    pickle_loaded = True
    return plist, bats, bowls

plist, bats, bowls = load_pickle()
def get_hubs_auth(params=None):
    # if pickle_loaded == False:
    hlist, alist = [], []
    print "Calculating weighing function"
    for pl in range(len(plist)):
        # print pl
        try:
            playermat = np.zeros((len(bowls)+len(bats), len(bowls)+len(bats)))
            for key, value in plist[pl].iteritems():
                a = (value["runs"]/value["balls"])
                b = (value["dots"]/value["balls"])
                c = value["matches"]
                d = value["sixes"]
                e = value["fours"]
                weighted_score = ((hits_alpha1*a) + (hits_alpha2*b) + (hits_alpha3*c) + (hits_alpha4*d) + (hits_alpha5*e))
                playermat[np.where(bowls==key[1])[0][0]][len(bowls)+np.where(bats==key[0])[0][0]] = weighted_score
            G = nx.from_numpy_matrix(playermat)
            h, a = nx.hits(G)
            hlist.append(h)
            alist.append(a)
        except:
            hlist.append([0])
            alist.append([0])
    return hlist, alist
    # with open('../data/player_hlist.pkl', 'w') as fp:
    #     pickle.dump(hlist, fp)

    # with open('../data/player_alist.pkl', 'w') as fp:
    #     pickle.dump(alist, fp)

    # with open('../data/batsmen.pkl', 'w') as fp:
    #     pickle.dump(bats, fp)

    # with open('../data/bowlers.pkl', 'w') as fp:
    #     pickle.dump(bowls, fp)

# def player_graph(params):
#     plist, bats, bowls = load_pickle()
#     plist = plist[-1]
#     playermat = [[0 for i in range(len(bowls)+len(bats))] for j in range(len(bowls)+len(bats))]
#     for key, value in plist.iteritems():
#         a = (value["runs"]/value["balls"])
#         b = (value["dots"]/value["balls"])
#         c = value["matches"]
#         d = value["sixes"]
#         e = value["fours"]
#         weighted_score = ((hits_alpha1*a) + (hits_alpha2*b) + (hits_alpha3*c) + (hits_alpha4*d) + (hits_alpha5*e))
#         playermat[np.where(bowls==key[1])[0][0]][len(bowls)+np.where(bats==key[0])[0][0]] = weighted_score/50.
    
#     G = nx.from_numpy_matrix(np.array(playermat))
#     h, a = nx.hits(G)
#     h = sorted(h, key=h.get, reverse=True)[:20]
#     a = sorted(a, key=a.get, reverse=True)[:20]
    
#     playfin = [[0 for i in range(40)] for j in range(40)]
#     for i in range(len(h)):
#         for j in range(len(a)):
#             print i, j, h[i], a[j]
#             playfin[i][len(h)+j] = playermat[h[i]][a[j]]
    
#     g = ig.Graph.Full(n = 40, directed = True)
#     g = g.Weighted_Adjacency(playfin, mode = "DIRECTED")
#     # g.vs["name"] = teams_id.keys()
#     g.delete_vertices(np.where(np.array(g.vs.degree())==0)[0])
#     save_graph(g)

# def save_graph(g, layoutname = "fruchterman_reingold"):
#     if layoutname == "kk":
#         g.write_svg("newgraph.svg", labels = "name" , layout = g.layout(layoutname))
#     else:
#         igraph.plot(g, 'newgraph.png', layout=g.layout(layoutname), bbox=(500, 500), margin= 50, hovermode='closest', edge_width = g.es["weight"], 
#             vertex_size=20, vertex_color = "red", vertex_label_dist = 2, vertex_label_size = 5, edge_arrow_size=1)


#player_graph()
#plt.bar(range(len(h)), h.values(), align='center')
#plt.xticks(range(len(h)), h.keys())
#plt.show(h)
# load_pickle()
# get_hubs_auth()
