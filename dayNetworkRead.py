import networkx as nx
import numpy as np
import pprint
import matplotlib.pyplot as plt
import scipy as sp

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

plt.imshow(A_M, cmap='hot', interpolation = 'nearest')
plt.show()

