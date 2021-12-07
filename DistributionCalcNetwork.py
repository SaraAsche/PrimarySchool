import networkx as nx
from networkx.algorithms.operators.binary import difference, disjoint_union, symmetric_difference
import numpy as np
import pprint
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import seaborn as sns
from matplotlib.colors import LogNorm
import math
import pickle
from sklearn.linear_model import LinearRegression
#import community as community_louvain

graph1 = nx.Graph() #day1
graph2 = nx.Graph() #day2

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

'''

with open('dayOneNewIndexLunch.csv', mode='r') as primarySchoolData1:

    for line in primarySchoolData1:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        #weightln = 1+np.log(int(temp[5]))
        graph1.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
        graph1.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
        graph1.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph1.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))

with open('dayTwoNewIndexLunch.csv', mode='r') as primarySchoolData2:

    for line in primarySchoolData2:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        #weightln = 1+np.log(int(temp[5]))
        graph2.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
        graph2.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
        #graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
    



for i in range(1,6):
    with open('dayOneNewIndexLunch10minClass'+str(i)+'.csv', mode='r') as primarySchoolData:
        graph =nx.Graph()
        for line in primarySchoolData:
            temp = list(map(lambda x: x.strip(), line.split(",")))
            
            graph.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
            graph.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
            graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
            graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
    if i ==1:
        graphDay1Class1 = graph
    elif i ==2:
        graphDay1Class2 = graph
    elif i ==3:
        graphDay1Class3 = graph
    elif i ==4:
        graphDay1Class4 = graph
    else:
        graphDay1Class5 = graph


for i in range(1,6):
    with open('dayTwoNewIndexPrimaryLunch10minClass' + str(i)+ '.csv', mode='r') as primarySchoolData:
        graph =nx.Graph()
        for line in primarySchoolData:
            temp = list(map(lambda x: x.strip(), line.split(",")))
            
            graph.add_nodes_from([(int(temp[1]), {'klasse' : temp[3]})])
            graph.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})])
            graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
            graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
    if i ==1:
        graphDay2Class1 = graph
    elif i ==2:
        graphDay2Class2 = graph
    elif i ==3:
        graphDay2Class3 = graph
    elif i ==4:
        graphDay2Class4 = graph
    else:
        graphDay2Class5 = graph
'''

'''
a_list = list(graphDay1Class1.nodes)
a_list.sort()

A = nx.adjacency_matrix(graphDay1Class1, nodelist=a_list)

'''
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
    if gradeInteraction: #Interactions between students in same grade but not class
        
        for node, klasseAttr in grades.items():
            for n in graph.neighbors(node):
                klasseAttr_neighbour = nx.get_node_attributes(graph, 'klasse')[n]
                if klasseAttr[0] == klasseAttr_neighbour[0] and klasseAttr[1] != klasseAttr_neighbour[1]:        
                    G.add_edge(node, n, weight=graph[node][n]['weight'])
                    G.add_node(n, klasse=graph.nodes[n]['klasse'])
            #if klasseAttr != 'Teachers':
             #   for n in graph.neighbors(node):
              #      klasseAttr_neighbour = nx.get_node_attributes(graph, 'klasse')[n]
               #     if klasseAttr_neighbour != 'Teachers' and klasseAttr[0] == klasseAttr_neighbour[0] and klasseAttr[1] != klasseAttr_neighbour[1]:        
                #        G.add_edge(node, n, weight=graph[node][n]['weight'])
                 #       G.add_node(n, klasse=graph.nodes[n]['klasse'])
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
        plt.ylabel('Normalised log frequency')
    else:
        plt.yscale('linear')
        plt.ylabel('Frequency')

    if logX:
        plt.xscale('log') 
        plt.xlabel('log Degree')
    else:
        plt.xscale('linear')
        plt.xlabel('Degree')

    #plt.plot(x,y, color = 'skyblue')
    plt.show()

def pixelDist(graph): 
    a_list = list(graph.nodes)
    a_list.sort()

    A = nx.adjacency_matrix(graph, nodelist=a_list)

    A_M = A.todense()

    return None

def makeHeatMap(subGraph):
    a_list = list(subGraph.nodes)
    a_list.sort()

    A = nx.adjacency_matrix(subGraph, nodelist=a_list)

    A_M = A.todense()

    ax = sns.heatmap(A_M, robust= False)
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

def generateSchoolGraph(graph):
    school = nx.Graph()
    sub = createSubGraphWithoutGraph(graph, True, True)

    for u, v, a in graph.edges(data = True):
        if not sub.has_edge(u,v):
            school.add_edge(u, v, count = a)
        else:
            school.add_edge(u, v, count = (a-sub[u][v].items()))
    return school


    

#graphDay1Class1

'''
histDistribution(graph1)
histDistribution(graph2)

histDistributionLog(graph1, False, True)
histDistributionLog(graph2, False, True)
'''
#BA = nx.barabasi_albert_graph(230, 10)
#histDistributionLog(BA, False, True)

#histDistributionLog(graph1, False, True)
#histDistributionLog(graph2, False, True)

'''
l = createSubGraphWithoutGraph(graph1, True, False)
makeHeatMap(l)
histDistributionLog(l, False, True)

p = createSubGraphWithoutGraph(graph2, True, True)
makeHeatMap(p)
histDistributionLog(p, False, True)
'''
def plot_Correlation_between_Days(day1, day2):

#graph1.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})]) graph1.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
    #degday1 = [val for (node, val) in sorted(day1.degree(weight = 'weight'))]
    #degday2 = [val for (node, val) in sorted(day2.degree(weight = 'weight'))]

    degday1 = sorted(day1.degree(weight = 'weight'))
    dd1 = []
    degday2 = sorted(day2.degree(weight = 'weight'))
    dd2= []

    for node, val in degday1:
        for node2, val2 in degday2:
            if node == node2:
                dd1.append(val)
                dd2.append(val2)

    plt.scatter(dd1, dd2)
    
    print("Pearson correlation:")
    print(np.corrcoef(dd1, dd2))
    print(stats.pearsonr(dd1, dd2))
    plt.show()

def checkSingleNodeDist(graph, logX, logY, node):
    
    degs=graph1.adj[node]
    
    data = list(map(lambda x: x['weight'], list(degs.values())))

    N = len(data)
    sorteddata = np.sort(data)

    d = toCumulative(sorteddata)

    plt.plot(d.keys(), d.values())

    if logY:
        plt.yscale('log') 
        plt.ylabel('Normalised log frequency')
    else:
        plt.yscale('linear')
        plt.ylabel('Frequency')

    if logX:
        plt.xscale('log') 
        plt.xlabel('log edge weight')
    else:
        plt.xscale('linear')
        plt.xlabel('Edge weight')

    #plt.plot(x,y, color = 'skyblue')
    plt.show()

def linRegOnenode(graph, node):
    degs=graph1.adj[node]
    
    data = list(map(lambda x: x['weight'], list(degs.values())))

    N = len(data)
    sorteddata = np.sort(data)

    d = toCumulative(sorteddata)
    x=d.keys()
    y=d.values()

    logx = np.fromiter((map(math.log10, x)), dtype=float).reshape((-1,1))
    logy = np.fromiter((map(math.log10, y)), dtype=float).reshape((-1,1))

    model = LinearRegression().fit(logx,logy)
    intercept = model.intercept_
    slope = model.coef_

    print(model.score(logx, logy))

    #plt.plot(logx, intercept + slope*logx)
    #plt.show()
    return float(slope)


def plotReg(graph):
    slopes = []
    degrees = []
    for node in list(graph.nodes):
        slopes.append(linRegOnenode(graph1, node))
        degrees.append(graph.degree(weight = 'weight')[node])
        #degrees.append(graph.degree()[node]) #only degree

    plt.scatter(degrees, slopes)


    degreesAr = np.array(degrees).reshape((-1,1))
    slopesAr = np.array(slopes).reshape((-1,1))

    model = LinearRegression().fit(degreesAr,slopesAr)
    intercept = model.intercept_
    slope = model.coef_

    plt.plot(degreesAr, intercept + slope*degreesAr)

    plt.xlabel('Degree')
    plt.ylabel('Slope a')

    plt.show()

    print("R^2 for samlet regresjon "+str(model.score(degreesAr, slopesAr)))


#makeHeatMap(graph2)

#histDistributionLog(graph1, False, True)
#histDistributionLog(graph2, False, True)

#linRegOnenode(graph1, 1)
#checkSingleNodeDist(graph1, True, True, 1)

plotReg(graph1)

'''
checkSingleNodeDist(graph1, True, True, 22)

print(max(dict(graph1.edges).items(), key=lambda x: x[1]['weight']))

print(sorted(graph1.degree(weight='weight'), key=lambda x: x[1], reverse=True))
'''
#plot_Correlation_between_Days(graph1, graph2)

#plot_degree_distribution(p)