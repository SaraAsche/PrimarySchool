from cProfile import label
from re import I
import networkx as nx
from networkx.algorithms.operators.binary import difference, disjoint_union, symmetric_difference
import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import seaborn as sns
from matplotlib.colors import LogNorm
import math
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.transforms import TransformedBbox
from itertools import groupby
import matplotx

# import community as community_louvain

graph1 = nx.Graph()  # day1
graph2 = nx.Graph()  # day2

with open("dayOneNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph1.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph1.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        graph1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

with open("dayTwoNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph2.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph2.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        # graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

"""

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
"""

"""
a_list = list(graphDay1Class1.nodes)
a_list.sort()

A = nx.adjacency_matrix(graphDay1Class1, nodelist=a_list)

"""
# A_M = A.todense()


def plot_degree_dist_log(G):
    m = 3
    degree_freq = nx.degree_histogram(G)
    degrees = range(len(degree_freq))
    plt.figure(figsize=(12, 8))
    plt.loglog(degrees[m:], degree_freq[m:], "go-")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.xlim(7)
    plt.show()


def plot_degree_distribution(G):
    degs = {}
    for n in G.nodes():
        deg = G.degree(n, weight="weight")
        degs[n] = deg

    items = sorted(degs.items())

    data = []
    for line in items:
        data.append(line[1])

    fig = plt.figure()

    values, base = np.histogram(data, bins=40)

    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c="skyblue")

    plt.title("Primary school degree distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()
    # fig.savefig("degree_distribution.png")


def histDistribution(graph):
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n, weight="weight")
        degs[n] = deg

    items = sorted(degs.items())

    data = []
    for line in items:
        data.append(line[1])

    plt.hist(data, bins=10, color="skyblue", ec="black")  # col = 'skyblue for day2, mediumseagreen for day1
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()


def createSubGraph(graph, grade, klasse):
    res = []
    grades = nx.get_node_attributes(graph, "klasse")
    res = list(
        map(
            lambda x: x[0],
            filter(lambda x: str(grade) in x[1] and (klasse in x[1] if klasse is not None else True), grades.items()),
        )
    )

    return graph.subgraph(res)


def createSubGraphWithout(
    graph, grade, klasse
):  # Function generate network without all interactions within the same grade. If klasse = true: without the class interactions
    res = []
    grades = nx.get_node_attributes(graph, "klasse")
    # res = list(map(lambda x: x[0], filter(lambda x: str(grade) in x[1] and (klasse in x[1] if klasse is not None else True), grades.items())))
    # print(type(list(grades.keys())[0]))
    G = nx.Graph()

    nodes = set()
    edges = []
    for node, klasseAttr in grades.items():
        for n in graph.neighbors(node):
            if grade and (not klasse):
                if grades[n][0] != grades[node][0]:
                    G.add_edge(node, n, weight=graph[node][n]["weight"])
                    G.add_node(n, klasse=graph.nodes[n]["klasse"])
            elif not grade:
                if grades[n] != grades[node]:
                    G.add_edge(node, n, weight=graph[node][n]["weight"])
                    G.add_node(n, klasse=graph.nodes[n]["klasse"])
    return G


def createSubGraphWithoutGraph(graph, diagonal, gradeInteraction):
    res = []
    G = nx.Graph()
    nodes = set()
    edges = []
    grades = nx.get_node_attributes(graph, "klasse")
    if diagonal:  # just interactions between the students in the same class
        for node, klasseAttr in grades.items():
            for n in graph.neighbors(node):
                if grades[n] == grades[node]:
                    G.add_edge(node, n, weight=graph[node][n]["weight"])
                    G.add_node(n, klasse=graph.nodes[n]["klasse"])
    if gradeInteraction:  # Interactions between students in same grade but not class

        for node, klasseAttr in grades.items():
            for n in graph.neighbors(node):
                klasseAttr_neighbour = nx.get_node_attributes(graph, "klasse")[n]
                if klasseAttr[0] == klasseAttr_neighbour[0] and klasseAttr[1] != klasseAttr_neighbour[1]:
                    G.add_edge(node, n, weight=graph[node][n]["weight"])
                    G.add_node(n, klasse=graph.nodes[n]["klasse"])
            # if klasseAttr != 'Teachers':
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
        cumul -= float(dictHist[i]) / float(n)
    return cHist


def histDistributionLog(graph, logX, logY, output=False, day1=True, wait=False, axis=None, old=False):
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n, weight="weight")
        degs[n] = deg
        # degs[n] = 1-np.log(deg)

    items = sorted(degs.items())

    data = []
    for line in items:
        data.append(line[1])

    N = len(data)
    sorteddata = np.sort(data)
    print(sorteddata)
    d = toCumulative(sorteddata)

    if axis:
        if not old:
            axis.plot(d.keys(), d.values(), label="Day 1")
        if old:
            axis.plot(d.keys(), d.values(), label="Day 2")
        if logX:
            axis.set_xscale("log")
            axis.set_xlabel("log Degree")
        else:
            axis.set_xscale("linear")
            axis.set_xlabel("Degree")
        if logY:
            axis.set_yscale("log")
            axis.set_ylabel("Normalised log frequency")
        else:
            axis.set_yscale("linear")
            axis.set_ylabel("Frequency")
    else:
        if day1:
            plt.plot(d.keys(), d.values(), label="Day 1", color="seagreen")
        else:
            plt.plot(d.keys(), d.values(), label="Day 2", color="coral")

        if logY:
            plt.yscale("log")
            plt.ylabel("Normalised log frequency")
        else:
            plt.yscale("linear")
            plt.ylabel("Frequency")

        if logX:
            plt.xscale("log")
            plt.xlabel("log Degree")
        else:
            plt.xscale("linear")
            plt.xlabel("Degree")

    # plt.plot(x,y, color = 'skyblue')
    if output:
        #####!!!
        return d
        return d.keys(), d.values()

    if not wait:
        plt.show()


def extractXYcoordinates(graph):
    x, y = histDistributionLog(graph, False, False, True)

    xList = list(x)
    yList = list(y)

    plt.scatter(xList, yList)

    degrees = np.array(xList).reshape((-1, 1))
    freq = np.array(yList)  # .reshape((1,-1))
    # freq = np.array(yList)

    return degrees, freq


def logRegression(graph):
    degrees, freq = extractXYcoordinates(graph)

    freq = (freq * 100).astype(int)
    # print(freq)

    # degreeLog = np.array(list(map(math.log10, degrees))).reshape((-1,1))
    # freqLog = np.array(list(map(math.log10, freq))).reshape((-1,1))

    model = LinearRegression()
    model.fit(degrees, freq)

    # sns.regplot(x=degrees, y=freq, logistic=True)

    intercept = model.intercept_
    slope = model.coef_

    print("intercept")
    print(intercept)
    print("coef")
    print(model.coef_)
    # print(model.predict(np.array([500]).reshape((1,-1))))

    # plt.plot(degrees, intercept + slope*degrees)

    # plt.show()

    """
    model = LogisticRegression().fit(degrees, freq)
    print(model.intercept_)
    print(model.coef_)
    print(model.score(degrees,freq))
    plt.show()
    """


def getGradeChanges(graph):
    d = {}

    for node in sorted(graph.nodes()):
        d[graph.nodes(data="klasse")[node]] = d.get(graph.nodes(data="klasse")[node], 0) + 1

    return list(d.values()), list(d.keys())


# def _get_text_object_bbox(text_obj, ax):
#     # https://stackoverflow.com/a/35419796/2912349
#     transform = ax.transData.inverted()
#     # the figure needs to have been drawn once, otherwise there is no renderer?
#     plt.ion(); plt.show(); plt.pause(0.001)
#     bb = text_obj.get_window_extent(renderer = ax.get_figure().canvas.renderer)
#     # handle canvas resizing
#     return TransformedBbox(bb, transform)

# def set_yrange_label(label, ymin, ymax, x, dx=-0.5, ax=None, *args, **kwargs):
#     """
#     Annotate a y-range.

#     Arguments:
#     ----------
#     label : string
#         The label.
#     ymin, ymax : float, float
#         The y-range in data coordinates.
#     x : float
#         The x position of the annotation arrow endpoints in data coordinates.
#     dx : float (default -0.5)
#         The offset from x at which the label is placed.
#     ax : matplotlib.axes object (default None)
#         The axis instance to annotate.
#     """

#     if not ax:
#         ax = plt.gca()

#     dy = ymax - ymin
#     props = dict(connectionstyle='angle, angleA=90, angleB=180, rad=0',
#                  arrowstyle='-',
#                  shrinkA=10,
#                  shrinkB=10,
#                  lw=1)
#     ax.annotate(label,
#                 xy=(x, ymin),
#                 xytext=(x + dx, ymin + dy/2),
#                 annotation_clip=False,
#                 arrowprops=props,
#                 *args, **kwargs,
#     )
#     ax.annotate(label,
#                 xy=(x, ymax),
#                 xytext=(x + dx, ymin + dy/2),
#                 annotation_clip=False,
#                 arrowprops=props,
#                 *args, **kwargs,
#     )


# def _get_text_object_bbox(text_obj, ax):
#     # https://stackoverflow.com/a/35419796/2912349
#     transform = ax.transData.inverted()
#     # the figure needs to have been drawn once, otherwise there is no renderer?
#     plt.ion(); plt.show(); plt.pause(0.001)
#     bb = text_obj.get_window_extent(renderer = ax.get_figure().canvas.renderer)
#     # handle canvas resizing
#     return TransformedBbox(bb, transform)

# def annotate_yranges(groups, ax=None):
#     """
#     Annotate a group of consecutive yticklabels with a group name.

#     Arguments:
#     ----------
#     groups : dict
#         Mapping from group label to an ordered list of group members.
#     ax : matplotlib.axes object (default None)
#         The axis instance to annotate.
#     """
#     if ax is None:
#         ax = plt.gca()

#     label2obj = {ticklabel.get_text() : ticklabel for ticklabel in ax.get_yticklabels()}
#     print(label2obj)
#     for ii, (group, members) in enumerate(groups.items()):
#         first = members[0]
#         last = str(int(members[-1]) + 1)

#         bbox0 = _get_text_object_bbox(label2obj[first], ax)
#         bbox1 = _get_text_object_bbox(label2obj[last], ax)

#         set_yrange_label(group, bbox0.y0 + bbox0.height/2,
#                          bbox1.y0 + bbox1.height/2,
#                          min(bbox0.x0, bbox1.x0),
#                          -2,
#                          ax=ax)


def test_table(A_M, grade_list, class_name):
    ids = [i for i in range(0, sum(grade_list) + 1)]
    seasons = []

    for name, count in zip(class_name, grade_list):
        seasons.extend([name] * count)

    print(len(ids))
    print(len(seasons))
    print(grade_list)
    tuples = list(zip(ids, seasons))
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

    new_A_M = []

    for elem in A_M:
        new_A_M.append(elem.getA1().tolist())

    d = dict(enumerate(new_A_M))

    df = pd.DataFrame(d, index=index)
    return df


def add_line(ax, xpos, ypos):
    # line = plt.Line2D([ypos + .03, ypos + .2], [xpos-.00199, xpos-.00199], linewidth=.8, color='dimgray', transform=ax.transAxes)
    line = plt.Line2D(
        [ypos + 0.03, ypos + 0.2],
        [xpos - 0.0037, xpos - 0.0037],
        linewidth=0.8,
        color="dimgray",
        transform=ax.transAxes,
        linestyle="--",
    )
    line.set_clip_on(False)
    ax.add_line(line)


def label_len(my_index, level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k, g in groupby(labels)]


def label_group_bar_table(ax, df):
    xpos = -0.2
    scale = 1.0 / df.index.size
    # for level in range(df.index.nlevels):
    pos = df.index.size
    for label, rpos in label_len(df.index, 1):
        print(label)
        if type(label) != int:
            add_line(ax, pos * scale, xpos)
            pos -= rpos
        lypos = (pos + 0.3 * rpos) * scale if label != "Teachers" else (pos - 0.3 * rpos) * scale
        ax.text(xpos + 0.1, lypos, label, ha="center", transform=ax.transAxes)
        # add_line(ax, pos*scale , xpos)
        # xpos -= .2


def makeHeatMap(subGraph, ax=None, output=False, wait=True):
    a_list = list(subGraph.nodes)
    a_list.sort()
    A = nx.adjacency_matrix(subGraph, nodelist=a_list)

    if output:
        return A

    A_M = A.todense()

    # norm = np.linalg.norm(A_M)

    # A_MN = A_M/norm

    grade_list, class_name = getGradeChanges(subGraph)
    print(grade_list)
    print(class_name)
    df = test_table(A_M, grade_list, class_name)
    fig = plt.figure(figsize=(6, 6))

    if not ax:
        ax = fig.add_subplot(111)
    # sns.heatmap(df, yticklabels=False,center=225, cmap = 'magma')
    sns.heatmap(df, yticklabels=False, robust=True, ax=ax)
    labels = ["" for _ in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    ax.set_ylabel("")

    label_group_bar_table(ax, df)
    fig.subplots_adjust(bottom=0.1 * df.index.nlevels)

    lst = [grade_list[0] + 1]

    for i in range(1, len(grade_list)):
        lst.append(grade_list[i] + lst[-1])

    ax.hlines(lst, *ax.get_xlim(), label=class_name, colors="dimgray", linewidth=0.8, linestyle="--")
    ax.vlines(lst, *ax.get_ylim(), label=class_name, colors="dimgray", linewidth=0.8, linestyle="--")

    # plt.savefig('Day2Heatmap.png', bbox_inches='tight', dpi=150)

    if wait:
        plt.show()


def pixelDist(graph, logY, logX, axis=None, output=False, old=False):
    A = makeHeatMap(graph, output=True)
    # print(A[np.triu_indices(236, k = 1)])
    length = len(graph.nodes())
    weights = A[np.triu_indices(length, k=1)].tolist()[0]

    data = sorted(weights)

    sorteddata = np.sort(data)
    d = toCumulative(sorteddata)

    if output:
        # return sorteddata
        return d

    if axis:
        if not old:
            axis.plot(d.keys(), d.values(), label="Day 1")
        if old:
            axis.plot(d.keys(), d.values(), label="Day 2")

        if logX:
            axis.set_xscale("log")
            axis.set_xlabel("log Degree")
        else:
            axis.set_xscale("linear")
            axis.set_xlabel("Frequency")
        if logY:
            axis.set_yscale("log")
            axis.set_ylabel("Normalised log frequency")
        else:
            axis.set_yscale("linear")
            axis.set_ylabel("Frequency")
    else:
        plt.plot(d.keys(), d.values())
        if logY:
            plt.yscale("log")
            plt.ylabel("Normalised log frequency")
        else:
            plt.yscale("linear")
            plt.ylabel("Frequency")
        if logX:
            plt.xscale("log")
            plt.xlabel("log Degree")
        else:
            plt.xscale("linear")
            plt.xlabel("Degree")
        plt.show()

    # print(np.triu(A.todense(), k=1)) #


def saveSubNetworks():
    withoutGrade_day1 = createSubGraphWithout(graph1, True, False)
    withoutGrade_day2 = createSubGraphWithout(graph2, True, False)

    onlyGrade_day1 = createSubGraphWithoutGraph(graph1, False, True)
    onlyGrade_day2 = createSubGraphWithoutGraph(graph2, False, True)

    nx.write_edgelist(onlyGrade_day1, "onlyGrade_day1.csv")
    nx.write_edgelist(onlyGrade_day2, "onlyGrade_day2.csv")
    nx.write_edgelist(withoutGrade_day1, "withoutGrade_day1.csv")
    nx.write_edgelist(withoutGrade_day2, "withoutGrade_day2.csv")


def saveNetwork(name, network):
    pickle.dump(network, open(name, "wb"))


def loadNetwork(name):
    return pickle.load(open(name, "rb"))


def generateSchoolGraph(graph):
    school = nx.Graph()
    sub = createSubGraphWithoutGraph(graph, True, True)

    for u, v, a in graph.edges(data=True):
        if not sub.has_edge(u, v):
            school.add_edge(u, v, count=a)
        else:
            school.add_edge(u, v, count=(a - sub[u][v].items()))
    return school


# graphDay1Class1

"""
histDistribution(graph1)
histDistribution(graph2)

histDistributionLog(graph1, False, True)
histDistributionLog(graph2, False, True)
"""
# BA = nx.barabasi_albert_graph(230, 10)
# histDistributionLog(BA, False, True)

# histDistributionLog(graph1, False, True)
# histDistributionLog(graph2, False, True)

"""
l = createSubGraphWithoutGraph(graph1, True, False)
makeHeatMap(l)
histDistributionLog(l, False, True)

p = createSubGraphWithoutGraph(graph2, True, True)
makeHeatMap(p)
histDistributionLog(p, False, True)
"""


def plot_Correlation_between_Days(day1, day2):

    # graph1.add_nodes_from([(int(temp[2]), {'klasse' : temp[4]})]) graph1.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
    # degday1 = [val for (node, val) in sorted(day1.degree(weight = 'weight'))]
    # degday2 = [val for (node, val) in sorted(day2.degree(weight = 'weight'))]

    degday1 = sorted(day1.degree(weight="weight"))
    dd1 = []
    degday2 = sorted(day2.degree(weight="weight"))
    dd2 = []

    for node, val in degday1:
        for node2, val2 in degday2:
            if node == node2:
                dd1.append(val)
                dd2.append(val2)

    plt.scatter(dd1, dd2, color="darkolivegreen")

    x = dd1
    y = dd2

    x = np.array(x).reshape((-1, 1))
    y = np.array(y).reshape((-1, 1))

    model = LinearRegression().fit(x, y)
    intercept = model.intercept_
    slope = model.coef_

    print(model.score(x, y))

    plt.plot(x, intercept + slope * x, color="black")
    plt.xlabel("Degree distribution on Day 1")
    plt.ylabel("Degree distribution on Day 2")

    print("Pearson correlation:")
    # print(np.corrcoef(dd1, dd2))
    r, p_val = stats.pearsonr(dd1, dd2)
    print(r)
    print(p_val)
    print(len(x))
    print(len(y))
    plt.savefig("DegreeDistributionAcrossDays.png", bbox_inches="tight", dpi=150)

    plt.show()


def checkSingleNodeDist(graph, logX, logY, node):

    degs = graph1.adj[node]

    data = list(map(lambda x: x["weight"], list(degs.values())))

    N = len(data)
    sorteddata = np.sort(data)

    d = toCumulative(sorteddata)

    plt.plot(d.keys(), d.values())

    if logY:
        plt.yscale("log")
        plt.ylabel("Normalised log frequency")
    else:
        plt.yscale("linear")
        plt.ylabel("Frequency")

    if logX:
        plt.xscale("log")
        plt.xlabel("log edge weight")
    else:
        plt.xscale("linear")
        plt.xlabel("Edge weight")

    # plt.plot(x,y, color = 'skyblue')
    plt.show()


def linRegOnenode(graph, node, class_class=False, grade_grade=False, off_diag=False, plot=False):
    if off_diag:
        lab = "Off diagonal: "
        graph_new = createSubGraphWithout(graph, True, False)
    elif class_class:
        lab = "Class-class: "
        graph_new = createSubGraphWithoutGraph(graph, True, False)
    elif grade_grade:
        lab = "Grade-grade: "
        graph_new = createSubGraphWithoutGraph(graph, False, True)
    else:
        lab = "Whole network: "
        graph_new = graph

    print(graph_new)
    degs = graph_new.adj[node]
    data = list(map(lambda x: x["weight"], list(degs.values())))

    N = len(data)
    sorteddata = np.sort(data)

    d = toCumulative(sorteddata)
    x = d.keys()
    y = d.values()

    logx = np.fromiter((map(math.log10, x)), dtype=float).reshape((-1, 1))
    logy = np.fromiter((map(math.log10, y)), dtype=float).reshape((-1, 1))

    model = LinearRegression().fit(logx, logy)
    intercept = model.intercept_
    slope = model.coef_

    score = float(model.score(logx, logy))

    if plot:
        lab += f"({score:.3})"
        plt.plot(logx, intercept + slope * logx, label=lab)
        plt.legend()
        plt.scatter(logx, logy)
        plt.xlabel("Log Degree")
        plt.ylabel("Log normalised frequency")
        # matplotx.line_labels()
        # plt.show()

    if not math.isnan(score):
        return score


def plotReg(graph):
    slopes = []
    degrees = []
    for node in list(graph.nodes):
        slopes.append(linRegOnenode(graph, node))
        degrees.append(graph.degree(weight="weight")[node])
        # degrees.append(graph.degree()[node]) #only degree

    plt.scatter(degrees, slopes)

    degreesAr = np.array(degrees).reshape((-1, 1))
    slopesAr = np.array(slopes).reshape((-1, 1))

    model = LinearRegression().fit(degreesAr, slopesAr)
    intercept = model.intercept_
    slope = model.coef_

    plt.plot(degreesAr, intercept + slope * degreesAr)

    plt.xlabel("Degree")
    plt.ylabel("Slope a")

    plt.show()

    print("R^2 for samlet regresjon " + str(model.score(degreesAr, slopesAr)))


def onlyTeacher(graph):
    teachers = []
    newGraph = nx.Graph()

    for node in graph.nodes(data=True):
        # print(node)
        if node[1]["klasse"] == "Teachers":
            teachers.append(node)

    for node in graph.nodes():
        for teacher in teachers:
            if node in graph.neighbors(teacher[0]):
                count = graph.get_edge_data(teacher[0], node)["weight"]
                newGraph.add_edge(teacher[0], node, weight=count)
    nx.draw(newGraph)
    plt.show()

    return newGraph


def heatmap_school(graph):
    off_diagonal = createSubGraphWithout(graph, True, False)
    grade_grade = createSubGraphWithoutGraph(graph, False, True)
    class_class = createSubGraphWithoutGraph(graph, True, False)

    # makeHeatMap(l)

    figure, axis = plt.subplots(2, 2, figsize=(10, 10))

    makeHeatMap(graph, axis[0, 0], wait=False)
    axis[0, 0].set_title("Whole network")
    makeHeatMap(off_diagonal, axis[1, 0], wait=False)
    axis[1, 0].set_title("Off diagonal")
    makeHeatMap(grade_grade, axis[0, 1], wait=False)
    axis[0, 1].set_title("Grade")
    makeHeatMap(class_class, axis[1, 1], wait=False)
    axis[1, 1].set_title("Class")

    figure.tight_layout()

    # plt.savefig('HeatmapSchool.png', bbox_inches='tight', dpi=150)

    plt.show()


def pickleDump(dictionary, name):
    graphFile = open(name, "wb")
    pickle.dump(dictionary, graphFile)
    graphFile.close()


def pixel_dist_school(graph, output=False, twoInOne=False, graph2=None):

    off_diagonal = createSubGraphWithout(graph, True, False)
    grade_grade = createSubGraphWithoutGraph(graph, False, True)
    class_class = createSubGraphWithoutGraph(graph, True, True)

    if output:
        wholeGraph = pixelDist(graph, True, True, output=True)
        pickleDump(wholeGraph, "graph2_whole_pixel.pkl")
        off_diag = pixelDist(off_diagonal, True, True, output=True)
        pickleDump(off_diag, "graph2_off_diag_pixel.pkl")
        grade = pixelDist(grade_grade, True, True, output=True)
        pickleDump(off_diag, "graph2_grade_pixel.pkl")
        classes = pixelDist(class_class, True, True, output=True)
        pickleDump(classes, "graph2_class_pixel.pkl")
        return None
    print("hey")
    figure, axis = plt.subplots(2, 2, figsize=(8, 8))
    figure.tight_layout(pad=4)

    pixelDist(graph, True, True, axis[0, 0])
    if twoInOne:
        pixelDist(graph2, True, True, axis[0, 0], old=True)
    axis[0, 0].set_title("Whole network")
    pixelDist(off_diagonal, True, True, axis[1, 0])
    if twoInOne:
        pixelDist(graph2, True, True, axis[1, 0], old=True)
    axis[1, 0].set_title("Off-diagonal")
    pixelDist(grade_grade, True, True, axis[0, 1])
    if twoInOne:
        pixelDist(graph1, True, True, axis[0, 1], old=True)
    axis[0, 1].set_title("grade-grade")
    pixelDist(class_class, True, True, axis[1, 1])
    if twoInOne:
        pixelDist(graph1, True, True, axis[1, 1], old=True)
    axis[1, 1].set_title("class-class")

    if twoInOne:
        handles, labels = axis[1, 1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center")

        plt.savefig("pixelDist_day1_day2.png", bbox_inches="tight", dpi=150)
        plt.show()

    # plt.savefig('pixelDistExperimental.png', bbox_inches='tight', dpi=150)
    else:
        plt.show()


def twoDayHeatmap(graph1, graph2):
    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

    makeHeatMap(graph1, ax=axis[0], wait=False)
    axis[0].set_title("a)", verticalalignment="bottom", y=-0.15)
    makeHeatMap(graph2, ax=axis[1], wait=False)
    axis[1].set_title("b)", verticalalignment="bottom", y=-0.15)
    figure.tight_layout()
    # plt.savefig('day1and2Heatmap.png',bbox_inches ='tight',dpi=150)

    plt.show()


def degreedist2days(graph1, graph2):
    histDistributionLog(graph1, logX=False, logY=True, output=False, wait=True)
    histDistributionLog(graph2, logX=False, logY=True, output=False, day1=False, wait=True)
    plt.show()


def cumulative2(graph):
    edges = graph.edges(data=True)
    weights = []

    for edge in edges:
        weights.append(edge[2]["weight"])

    count, bins_count = np.histogram(weights, bins=5)

    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    # plotting PDF and CDF
    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")

    plt.plot(bins_count[1:], cdf)
    plt.yscale("log")
    plt.ylabel("Normalised log frequency")
    plt.xscale("log")
    plt.xlabel("log Degree")

    plt.show()


def extractDictionaries(graph):
    pixel_dist_school(graph, output=True)
    dictionary = histDistributionLog(graph, logX=False, logY=True, output=True)
    pickleDump(dictionary, "Degreedistribution_Day1.pkl")
    dictionary = histDistributionLog(graph, logX=False, logY=True, output=True)
    pickleDump(dictionary, "Degreedistribution_Day2.pkl")


def school_hist_distribution(graph, logX=False, logY=True, twoInOne=False, graph2=None, output=False):
    off_diagonal = createSubGraphWithout(graph, True, False)
    grade_grade = createSubGraphWithoutGraph(graph, False, True)
    class_class = createSubGraphWithoutGraph(graph, True, True)

    if graph2:
        off_diagonal2 = createSubGraphWithout(graph2, True, False)
        grade_grade2 = createSubGraphWithoutGraph(graph2, False, True)
        class_class2 = createSubGraphWithoutGraph(graph2, True, True)

    if output:
        graph1 = histDistributionLog(graph, True, False, output=True)
        graph2 = histDistributionLog(graph2, True, False, output=True)
        off_diagonal1 = histDistributionLog(off_diagonal, True, False, output=True)
        off_diagonal2 = histDistributionLog(off_diagonal2, True, False, output=True)
        grade_grade1 = histDistributionLog(grade_grade, True, False, output=True)
        grade_grade2 = histDistributionLog(grade_grade2, True, False, output=True)
        class_class1 = histDistributionLog(class_class, True, False, output=True)
        class_class2 = histDistributionLog(class_class2, True, False, output=True)

        pickleDump(graph1, "DegreeDictwhole1.pkl")
        pickleDump(graph2, "DegreeDictwhole2.pkl")
        pickleDump(off_diagonal1, "DegreeDictOffDiag1.pkl")
        pickleDump(off_diagonal2, "DegreeDictOffDiag2.pkl")
        pickleDump(grade_grade1, "DegreeDictgrade1.pkl")
        pickleDump(grade_grade2, "DegreeDictgrade2.pkl")
        pickleDump(class_class1, "DegreeDictclass1.pkl")
        pickleDump(class_class2, "DegreeDictclass2.pkl")

        return "Da er alle lagt til"

    figure, axis = plt.subplots(2, 2, figsize=(8, 8))
    figure.tight_layout(pad=4)

    histDistributionLog(graph, logX, logY, wait=True, axis=axis[0, 0], day1=False)
    if twoInOne:
        histDistributionLog(graph2, logX, logY, axis=axis[0, 0], wait=True, old=True, day1=False)
    axis[0, 0].set_title("Whole network")
    histDistributionLog(off_diagonal, logX, logY, wait=True, axis=axis[1, 0], day1=False)
    if twoInOne:
        histDistributionLog(off_diagonal2, logX, logY, axis=axis[1, 0], wait=True, old=True, day1=False)
    axis[1, 0].set_title("Off diagonal")
    histDistributionLog(grade_grade, logX, logY, wait=True, axis=axis[0, 1], day1=False)
    if twoInOne:
        histDistributionLog(grade_grade2, logX, logY, axis=axis[0, 1], old=True, wait=True, day1=False)
    axis[0, 1].set_title("Grade-grade")
    histDistributionLog(class_class, logX, logY, wait=True, axis=axis[1, 1], day1=False)
    if twoInOne:
        histDistributionLog(class_class2, logX, logY, axis=axis[1, 1], old=True, wait=True, day1=False)
    axis[1, 1].set_title("Class-class")

    if not output:
        plt.savefig("DegreedistSubgroupsDay1And2.png", bbox_inches="tight", dpi=150)
        plt.show()


def outlierDist(graph) -> None:
    """Get the distribution of all interactions a max node has with others

    Parameters
    ----------
    graph : nx.Graph
        nx.Graph object that contains interactions at a primary school between Person objects.
    """

    dictionary = {}

    for (node, val) in graph.degree(weight="weight"):
        dictionary[node] = val

    sortedDegrees = dict(sorted(dictionary.items(), key=lambda item: item[1]))

    highest_degree_node = max(dictionary, key=dictionary.get)
    highest_degree = sortedDegrees[highest_degree_node]
    print(f"Highest degree node is {highest_degree_node} and its degree is {highest_degree}")

    list_of_interactions = []
    list_of_same_class = []
    list_of_same_grade = []

    for edge in graph.edges(highest_degree_node, data="weight"):

        list_of_interactions.append(edge[2])

        klasse = nx.get_node_attributes(graph, "klasse")

        i = edge[0]
        j = edge[1]

        if klasse[i] == klasse[j]:
            list_of_same_class.append(edge[2])
        if klasse[i][0] == klasse[j][0]:
            list_of_same_grade.append(edge[2])

    list_of_interactions.sort()
    print(list_of_interactions)
    print(f"length: {len(list_of_interactions)}")
    print(f"Average interaction: {Average(list_of_interactions)}")

    print(f"Interactions within same grade: {len(list_of_same_grade)}")
    print(f"Same grade average interaction: {Average(list_of_same_grade)}")

    print(f"Interactions within same class: {len(list_of_same_class)}")
    print(f"Same Class average interaction: {Average(list_of_same_class)}")


def Average(lst):
    return sum(lst) / len(lst)


def powerlawCheck(graph):
    R_whole = []
    R_off_diag = []
    R_grade_grade = []
    R_class_class = []

    off_diag = createSubGraphWithout(graph, True, False)
    class_class = createSubGraphWithoutGraph(graph, True, False)
    grade_grade = createSubGraphWithoutGraph(graph, False, True)

    for node in graph.nodes():
        R_whole.append(linRegOnenode(graph, node))
        if node in off_diag.nodes():
            R = linRegOnenode(graph, node, off_diag=True)
            if not (R == None):
                R_off_diag.append(R)
        if node in grade_grade.nodes():
            R = linRegOnenode(graph, node, grade_grade=True)
            if not (R == None):
                R_grade_grade.append(R)
        if node in class_class.nodes():
            R = linRegOnenode(graph, node, class_class=True)
            if not (R == None):
                R_class_class.append(R)

    print(R_grade_grade)
    print(R_class_class)
    print("-----------------")
    print(
        f"R2 for whole: {Average(R_whole)}, R2 for off-diag: {Average(R_off_diag)}, R2 for grade-grade: {Average(R_grade_grade)}, R2 for class-class: {Average(R_class_class)}"
    )
    print(
        f"whole {np.std(R_whole)}, off-diag {np.std(R_whole)}, grade {np.std(R_grade_grade)}, class {np.std(R_class_class)}"
    )
    print("-----------------")


def plotR(graph, node):
    off_diag = createSubGraphWithout(graph, True, False)
    class_class = createSubGraphWithoutGraph(graph, True, False)
    grade_grade = createSubGraphWithoutGraph(graph, False, True)

    linRegOnenode(graph, node, plot=True)
    if node in off_diag.nodes():
        linRegOnenode(graph, node, plot=True, off_diag=True)
    if node in grade_grade.nodes():
        linRegOnenode(graph, node, plot=True, grade_grade=True)
    if node in class_class:
        linRegOnenode(graph, node, plot=True, class_class=True)
    plt.savefig(f"./R2_img/{node}_R2.png")
    plt.clf()


def plotAllR(graph):

    for node in graph.nodes():
        plotR(graph, node)


# school_hist_distribution(graph1, output=True, graph2=graph2)

# pixel_dist_school(graph1, twoInOne=True, graph2=graph2)

# linRegOnenode(graph1, 0)

# outlierDist(graph1)
plotAllR(graph1)
# plotReg(graph1)
