import networkx as nx
import numpy as np
import pprint
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from matplotlib.colors import LogNorm
import math
#import community as community_louvain

graph = nx.Graph()
with open('dayTwoNewIndex.csv', mode='r') as primarySchoolData:
    

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        weightln = 1+np.log(int(temp[5]))
        graph.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
        graph.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
        #graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph.add_edge(int(temp[1]), int(temp[2]), weight = weightln)
        graph.add_edge(int(temp[1]), int(temp[2]), weight = weightln)

#nx.draw(graph)
#plt.show(block=True)

a_list = list(graph.nodes)
a_list.sort()

A = nx.adjacency_matrix(graph, nodelist=a_list)

A_M = A.todense()

ax = sns.heatmap(A_M)
#ax = sns.heatmap(A_M, robust = True)

#plt.show()
#plt.imshow(A_M, cmap='hot', interpolation = 'nearest')
#sns.set_theme(color_codes=True)
#A_M_S = np.argsort(np.argsort(A_M, axis=1), axis=1)
#g = sns.clustermap(A_M, figsize = (7,7), cbar_pos=(0, .2, .03, .4), vmin=1, vmax = 50)

#plot1 = plt.figure(1)
#g = sns.clustermap(A_M, figsize = (7,7), cbar_pos=(0, .2, .03, .4), robust=True, row_cluster=True, col_cluster= True)
#g = sns.clustermap(A_M, figsize = (7,7), cbar_pos=(0, .2, .03, .4))
#axes = g.data2d #gir ut pandas, skriv .axes for å få liste, hent ut 0 og 1. 

#n = sp.cluster.hierarchy.linkage(A_M)

#x1 = g.data2d.axes[0]
#x0 = list(range(len(x1)))
#plot2 = plt.figure(2)
#plt.scatter(x0, x1, marker='o')
plt.show()




