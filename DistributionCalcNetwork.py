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
        #weightln = 1+np.log(int(temp[5]))
        graph.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
        graph.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
        #graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))


a_list = list(graph.nodes)
a_list.sort()

A = nx.adjacency_matrix(graph, nodelist=a_list)

#A_M = A.todense()

def plot_degree_dist_log(G):
    m=3
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(12, 8)) 
    plt.loglog(degrees[m:], degree_freq[m:],'go-') 
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.xlim(7)
    plt.show()

def plot_degree_distribution(G):
    degs = {}
    for n in G.nodes ():
        deg = G.degree(n, weight='weight')
        degs[n] = deg

    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])

    fig = plt.figure()

    values, base = np.histogram(data, bins=40)
    
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='skyblue')
    
    plt.title("Primary school degree distribution")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()
    #fig.savefig("degree_distribution.png")

def histDistribution(graph):
    degs = {}
    for n in graph.nodes ():
        deg = graph.degree(n, weight='weight')
        degs[n] = deg

    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])

    plt.hist(data, bins=10, color='skyblue', ec = 'black') #col = 'skyeblue for day2, mediumseagreen for day1
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

def createSubGraph(graph, grade, klasse):
    res = []
    grades = nx.get_node_attributes(graph, 'klasse')
    res = list(map(lambda x: x[0], filter(lambda x: str(grade) in x[1] and (klasse in x[1] if klasse is not None else True), grades.items())))
    
    return graph.subgraph(res)


#histDistribution(graph)

#plot_degree_distribution(graph)


S = createSubGraph(graph, 1, 'B')

plot_degree_distribution(S)
