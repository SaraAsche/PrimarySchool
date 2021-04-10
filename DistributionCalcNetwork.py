import networkx as nx
import numpy as np
import pprint
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from matplotlib.colors import LogNorm
import math
import pickle
#import community as community_louvain

graph1 = nx.Graph()
graph2 = nx.Graph()
with open('dayOneNewIndex.csv', mode='r') as primarySchoolData:
    

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        #weightln = 1+np.log(int(temp[5]))
        graph1.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
        graph1.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
        graph1.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph1.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))

with open('dayTwoNewIndex.csv', mode='r') as primarySchoolData:
    

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        #weightln = 1+np.log(int(temp[5]))
        graph2.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
        graph2.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
        #graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))

a_list = list(graph1.nodes)
a_list.sort()

A = nx.adjacency_matrix(graph1, nodelist=a_list)

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
    print(data)
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

    plt.hist(data, bins=10, color='skyblue', ec = 'black') #col = 'skyblue for day2, mediumseagreen for day1
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()

def createSubGraph(graph, grade, klasse):
    res = []
    grades = nx.get_node_attributes(graph, 'klasse')
    res = list(map(lambda x: x[0], filter(lambda x: str(grade) in x[1] and (klasse in x[1] if klasse is not None else True), grades.items())))
    
    return graph.subgraph(res)

def createSubGraphWithout(graph, grade, klasse): #Function generate network without all interactions within the same grade. If klasse = true: without the class interactions
    res =[]
    print(grade and not klasse)
    grades = nx.get_node_attributes(graph, 'klasse')
    # res = list(map(lambda x: x[0], filter(lambda x: str(grade) in x[1] and (klasse in x[1] if klasse is not None else True), grades.items())))
    # print(type(list(grades.keys())[0]))
    print(graph.nodes[206])
    G = nx.Graph()

    nodes = set()
    edges = []
    for node, klasseAttr in grades.items():
        for n in graph.neighbors(node):
            if grade and (not klasse):
                if grades[n][0] != grades[node][0]:
                    G.add_edge(node, n, weight=graph[node][n]['weight'])
                    G.add_node(n, klasse=graph.nodes[n]['klasse'])
            elif not grade:
                if grades[n] != grades[node]:
                    G.add_edge(node, n, weight=graph[node][n]['weight'])
                    G.add_node(n, klasse=graph.nodes[n]['klasse'])
    return G

def createSubGraphWithoutGraph(graph, diagonal, gradeInteraction): 
    res =[]
    G = nx.Graph()
    nodes = set()
    edges = []
    grades = nx.get_node_attributes(graph, 'klasse')
    if diagonal: #just interactions between the students in the same class
        for node, klasseAttr in grades.items():
            for n in graph.neighbors(node):
                if grades[n] == grades[node]:
                    G.add_edge(node, n, weight=graph[node][n]['weight'])
                    G.add_node(n, klasse=graph.nodes[n]['klasse'])
    elif gradeInteraction: #Interactions between students in same grade but not class
        
        for node, klasseAttr in grades.items():
            if klasseAttr != 'Teachers':
                for n in graph.neighbors(node):
                    klasseAttr_neighbour = nx.get_node_attributes(graph, 'klasse')[n]
                    if klasseAttr_neighbour != 'Teachers' and klasseAttr[0] == klasseAttr_neighbour[0] and klasseAttr[1] != klasseAttr_neighbour[1]:        
                        G.add_edge(node, n, weight=graph[node][n]['weight'])
                        G.add_node(n, klasse=graph.nodes[n]['klasse'])
    return G

def toCumulative(l):                                                                                                
    n = len(l)                                                                                                      
    dictHist = {}                                                                                                   
    for i in l:                                                                                                     
        if i not in dictHist:                                                                                       
            dictHist[i] = 1                                                                                         
        else:                                                                                                       
            dictHist[i] += 1                                                                                        
    cHist = {}                                                                                                      
    cumul = 1                                                                                                       
    for i in dictHist:                                                                                              
        cHist[i] = cumul                                                                                            
        cumul -= float(dictHist[i])/float(n)
    return cHist
   
def histDistributionLog(graph, logX, logY):
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n, weight='weight')
        degs[n] = deg
        #degs[n] = 1-np.log(deg)

    items = sorted(degs.items())
    
    data = []
    for line in items:
        data.append(line[1])

    N = len(data)
    sorteddata = np.sort(data)
    d = toCumulative(sorteddata)

    print(float(sum(sorteddata))/float(len(sorteddata)))

    plt.plot(d.keys(), d.values())

    if logY:
        plt.yscale('log') 
    else:
        plt.yscale('linear')

    if logX:
        plt.xscale('log') 
    else:
        plt.xscale('linear')

    plt.xlabel('Degree')
    plt.ylabel('Normalised log frequency')

    #plt.plot(x,y, color = 'skyblue')
    plt.show()

def makeHeatMap(subGraph):
    a_list = list(subGraph.nodes)
    a_list.sort()

    A = nx.adjacency_matrix(subGraph, nodelist=a_list)

    A_M = A.todense()

    ax = sns.heatmap(A_M, robust= True)
    plt.show()

def saveSubNetworks():
    withoutGrade_day1 = createSubGraphWithout(graph1, True, False)
    withoutGrade_day2 = createSubGraphWithout(graph2, True, False)

    onlyGrade_day1 = createSubGraphWithoutGraph(graph1, False, True)
    onlyGrade_day2 = createSubGraphWithoutGraph(graph2, False, True)

    nx.write_edgelist(onlyGrade_day1, 'onlyGrade_day1.csv')
    nx.write_edgelist(onlyGrade_day2, 'onlyGrade_day2.csv') 
    nx.write_edgelist(withoutGrade_day1, 'withoutGrade_day1.csv')
    nx.write_edgelist(withoutGrade_day2, 'withoutGrade_day2.csv')


def saveNetwork(name, network):
    pickle.dump(network, open(name, 'wb'))

def loadNetwork(name):
    return pickle.load(open(name, 'rb'))

withoutGrade_day2 = createSubGraphWithout(graph2, True, False)

onlyGrade_day1 = createSubGraphWithoutGraph(graph1, False, True)
onlyGrade_day2 = createSubGraphWithoutGraph(graph2, False, True)

# withoutGrade_day1 = createSubGraphWithout(graph1, True, False)
saveNetwork('withoutGrade_day2', withoutGrade_day2)
saveNetwork('onlyGrade_day1', onlyGrade_day1)
saveNetwork('onlyGrade_day2', onlyGrade_day2)



#BA = nx.barabasi_albert_graph(230, 10)
#histDistributionLog(BA, False, True)