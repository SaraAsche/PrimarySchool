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
with open('dayOnePrimary.csv', mode='r') as primarySchoolData:
    
    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        graph.add_nodes_from([(temp[1], {'klasse' : temp[3]})])
        graph.add_nodes_from([(temp[2], {'klasse' : temp[4]})])
        graph.add_edge(temp[1], temp[2], weight = int(temp[5]))
        graph.add_edge(temp[1], temp[2])

#nx.draw(graph)
#plt.show(block=True)

A = nx.adjacency_matrix(graph)

A_M = A.todense()

#ax = sns.heatmap(A_M, vmin=1, vmax = 100, robust = True)
#plt.show()
#plt.imshow(A_M, cmap='hot', interpolation = 'nearest')
#sns.set_theme(color_codes=True)
#A_M_S = np.argsort(np.argsort(A_M, axis=1), axis=1)
#A_M_S2 = np.argsort(np.argsort(A_M_S, axis=-1),axis =-1)
g = sns.clustermap(A_M, figsize = (7,7), cbar_pos=(0, .2, .03, .4), vmin=1, vmax = 100)
plt.show()



