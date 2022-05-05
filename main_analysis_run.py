"""main_analysis_run.py

A file that contains most of the main analysis that has been run on the empiric primary school data. 
Each nx.Graph objects is generated each time the file is run, based on an accumulated interaction csv generated in dataPreperation.py

Author: Sara Johanne Asche
Date: 05.05.2022
File: main_analysis_run.py
"""


import matplotx
import networkx as nx

from networkx.algorithms import community

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

import math
import pickle
from sklearn.linear_model import LinearRegression
from itertools import groupby


graph1 = nx.Graph()  # day1
graph2 = nx.Graph()  # day2

# Generating a nx.Graph for day 1
with open("./data/dataPreperation/dayOneNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph1.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph1.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        graph1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

# Generating a nx.Graph for day 2
with open("./data/dataPreperation/dayTwoNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph2.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph2.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        # graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))


graph_lunch_1 = nx.Graph()
graph_lunch_2 = nx.Graph()

# Generating a nx.Graph for lunch on day 1
with open("./data/dayOneNewIndexLunch.csv", mode="r") as primarySchoolData1:

    for line in primarySchoolData1:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph_lunch_1.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph_lunch_1.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        graph_lunch_1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph_lunch_1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

# Generating a nx.Graph for lunch on day 2
with open("./data/dayTwoNewIndexLunch.csv", mode="r") as primarySchoolData2:

    for line in primarySchoolData2:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph_lunch_2.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph_lunch_2.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        # graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph_lunch_2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph_lunch_2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))


def createSubGraphWithout(graph: nx.Graph, grade: bool, klasse: bool) -> nx.Graph:
    """Generates a subgraph of a nx.Graph object.

    Function generate network without all interactions within the same grade. If klasse = true: without the class interactions

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    grade : bool
        Denotes whether or not grade interactions should be included
    klasse : bool
        Denotes whether or not grade interactions should be included
    """

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


def createSubGraphWithoutGraph(graph: nx.Graph, diagonal: bool, gradeInteraction: bool) -> nx.Graph:
    """Generates a subgraph of a nx.Graph object

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    diagonal : bool
        Denotes whether or not off-diagonal interactions should be included
    gradeInteraction : bool
        Denotes whether or not grade interactions should be included
    """
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
    return G


def toCumulative(l: list) -> dict:
    """Takes in a list of numbers and returns a dict with their frequency as key of number as val

    Parameters
    ----------
    l : list
        List containing ints
    """
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


def histDistributionLog(graph: nx.Graph, logX: bool, logY: bool, output=False, day1=True, wait=False, axis=None):
    """Generates the degree distribution of a nx.Graph object

    Parameters
    ----------
    G : nx.Graph
        A nx.Graph object that describes interactions in the network
    logY : bool
        Denotes whether or not y-axis should be log-scaled
    logX : bool
        Denotes whether or not x-axis should be log-scaled
    output : bool
        Denotes whether or not the dict of frequencies of interactions should be returned. Default is True
    day1 : bool
        Denote if histdist is run on day1 or day2. Default is True.
    wait : bool
        Whether or not to plot or wait for more curves to be drawn on same figure before plotting. Default is False
    axis : matplotlib.axis or None
        Describes where the interaction should be plotted on multiple grid figure
    """
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n, weight="weight")
        degs[n] = deg
        # degs[n] = 1-np.log(deg)

    items = sorted(degs.items())

    data = []
    for line in items:
        data.append(line[1])

    sorteddata = np.sort(data)
    print(sorteddata)
    d = toCumulative(sorteddata)

    if axis:
        if not day1:
            axis.plot(d.keys(), d.values(), label="Day 1")
        if day1:
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
            plt.plot(d.keys(), d.values(), label="Day 1", color="rosybrown")
        else:
            plt.plot(d.keys(), d.values(), label="Day 2", color="cadetblue")

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

    if output:
        return d.keys(), d.values()

    if not wait:
        plt.legend()
        plt.savefig("./fig_master/degree_dist_heatmap.png", dpi=500)
        plt.show()


def getGradeChanges(graph: nx.Graph) -> list:
    """Function to extract when grades change on heatmap for makeheatmap function"""
    d = {}

    for node in sorted(graph.nodes()):
        d[graph.nodes(data="klasse")[node]] = d.get(graph.nodes(data="klasse")[node], 0) + 1

    return list(d.values()), list(d.keys())


def test_table(A_M: np.Array, grade_list: list, class_name: list) -> pd.DataFrame:
    """Function to return a dataframe of an adjacency matric for a given graph to aid makeheatmap function"""
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
    """Function to add lines between grades to aid makeheatmap function"""
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
    """Function to return group labels to aid makeheatmap function"""
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k, g in groupby(labels)]


def label_group_bar_table(ax, df):
    """Function to group labels to aid makeheatmap function"""
    xpos = -0.2
    scale = 1.0 / df.index.size

    pos = df.index.size
    for label, rpos in label_len(df.index, 1):
        print(label)
        if type(label) != int:
            add_line(ax, pos * scale, xpos)
            pos -= rpos
        lypos = (pos + 0.3 * rpos) * scale if label != "Teachers" else (pos - 0.3 * rpos) * scale
        ax.text(xpos + 0.1, lypos, label, ha="center", transform=ax.transAxes)


def makeHeatMap(graph: nx.Graph, ax=None, output=False, wait=True):
    """Function to generate a heatmap of interactions with drawn lines for class separation

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    axis : matplotlib.axis or None
        Describes where the interaction should be plotted on multiple grid figure
    output : bool
        Denotes whether or not the dict of frequencies of interactions should be returned. Default is False
    wait : bool
        Whether or not to plot or wait for more curves to be drawn on same figure before plotting. Default is True
    """
    a_list = list(graph.nodes)
    a_list.sort()
    A = nx.adjacency_matrix(graph, nodelist=a_list)

    if output:
        return A

    A_M = A.todense()

    grade_list, class_name = getGradeChanges(graph)
    df = test_table(A_M, grade_list, class_name)
    fig = plt.figure(figsize=(6, 6))

    if not ax:
        ax = fig.add_subplot(111)

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

    plt.savefig("./fig_master/Day2Heatmap.png", bbox_inches="tight", dpi=150)

    if wait:
        plt.show()


def pixelDist(graph: nx.Graph, logY: bool, logX: bool, axis=None, output=False, old=False):
    """Interaction/pixel distribution of a nx.Graph object

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    logY : bool
        Denotes whether or not y-axis should be log-scaled
    logX : bool
        Denotes whether or not x-axis should be log-scaled
    axis : matplotlib.axis or None
        Describes where the interaction should be plotted on multiple grid figure
    output : bool
        Denotes whether or not the dict of frequencies of interactions should be returned. Default is True
    wait : bool
        Whether or not to plot or wait for more curves to be drawn on same figure before plotting. Default is True
    """
    A = makeHeatMap(graph, output=True)
    length = len(graph.nodes())
    weights = A[np.triu_indices(length, k=1)].tolist()[0]

    data = sorted(weights)

    sorteddata = np.sort(data)
    d = toCumulative(sorteddata)

    if output:
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


def saveSubNetworks() -> None:
    """Saves the edgelist of the different layers of the empiric model"""
    withoutGrade_day1 = createSubGraphWithout(graph1, True, False)
    withoutGrade_day2 = createSubGraphWithout(graph2, True, False)

    onlyGrade_day1 = createSubGraphWithoutGraph(graph1, False, True)
    onlyGrade_day2 = createSubGraphWithoutGraph(graph2, False, True)

    nx.write_edgelist(onlyGrade_day1, ".data/onlyGrade_day1.csv")
    nx.write_edgelist(onlyGrade_day2, ".data/onlyGrade_day2.csv")
    nx.write_edgelist(withoutGrade_day1, ".data/withoutGrade_day1.csv")
    nx.write_edgelist(withoutGrade_day2, ".data/withoutGrade_day2.csv")


def saveNetwork(name, network) -> None:
    """Saves the nx.Graph object network by using pickle"""
    pickle.dump(network, open(name, "wb"))


def loadNetwork(name) -> nx.Graph:
    """Loads the nx.Graph object network by using pickle"""
    return pickle.load(open(name, "rb"))


def plot_Correlation_between_Days(day1: nx.Graph, day2: nx.Graph) -> None:
    """Plots the linear correlation between the nx.Graph objects day1 and day2"""
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
    print(np.corrcoef(dd1, dd2))
    r, p_val = stats.pearsonr(dd1, dd2)
    print(r)
    print(p_val)
    print(len(x))
    print(len(y))
    plt.savefig("./fig_master/DegreeDistributionAcrossDays.png", bbox_inches="tight", dpi=150)
    plt.show()


def checkSingleNodeDist(graph: nx.Graph, logX: bool, logY: bool, node: bool) -> None:
    """Plots the weight distribution of single noded

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    logY : bool
        Denotes whether or not y-axis should be log-scaled
    logX : bool
        Denotes whether or not x-axis should be log-scaled
    node : int
        The ID of the specific node to plot interactions for
    """

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

    plt.show()


def linRegOnenode(graph: nx.Graph, node: int, class_class=False, grade_grade=False, off_diag=False, plot=False):
    """Draws linear regression for a given node in a nx.Graph object

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    node : int
        The ID of the specific node to plot interactions for
    class_class : bool
        Whether or not to only use interactions within the same class
    grade_grade : bool
        Whether or not to only use interactions within the same grade
    off-diag : bool
        Whether or not to only use interactions within the same class
    plot : bool
        whether to plot the result or return the R2 value
    """
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
        matplotx.line_labels()
        plt.show()

    if not math.isnan(score):
        return score


def plotReg(graph: nx.Graph):
    """Draws linear regression for a all nodes in a nx.Graph object

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    """
    slopes = []
    degrees = []
    for node in list(graph.nodes):
        slopes.append(linRegOnenode(graph, node))
        degrees.append(graph.degree(weight="weight")[node])

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


def heatmap_school(graph: nx.Graph):
    """Generates a heatmap for all layers of the school

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    """
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


def pickleDump(dictionary: dict, name: str) -> None:
    """Saves a dictionary through the pickle library

    Parameters
    ----------
    dictionary : dict
        dict of frequencies of values as keys and values as values
    name : str
        name in which the file should be saved as
    """
    graphFile = open(name, "wb")
    pickle.dump(dictionary, graphFile)
    graphFile.close()


def pixel_dist_school(graph: nx.Graph, output=False, twoInOne=False, graph2=None):
    """Draws a pixel/interaction distribution for either one or two nx.Graph objects

     Parameters
     ----------
     graph : nx.Graph
         A nx.Graph object that describes interactions in the network
    output : bool
         Whether or not to save the dicts as pickle files. Default is False
     twoInOne : bool
         Whether or not to include another day in the plot. Default is False
     graph2 : nx.Graph
         A second nx.Graph object that describes interactions in the network
    """

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

        plt.savefig("./fig_master/pixelDist_day1_day2.png", bbox_inches="tight", dpi=150)
        plt.show()

    else:
        plt.savefig("pixelDistExperimental.png", bbox_inches="tight", dpi=150)
        plt.show()


def twoDayHeatmap(graph1: nx.Graph, graph2: nx.Graph) -> None:
    """Generates heatmaps side by side for two nx.Graph objects

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    graph2 : nx.Graph
        A second nx.Graph object that describes interactions in the network
    """

    figure, axis = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

    makeHeatMap(graph1, ax=axis[0], wait=False)
    axis[0].set_title("a)", verticalalignment="bottom", y=-0.15)
    makeHeatMap(graph2, ax=axis[1], wait=False)
    axis[1].set_title("b)", verticalalignment="bottom", y=-0.15)
    figure.tight_layout()
    plt.savefig("day1and2Heatmap.png", bbox_inches="tight", dpi=150)
    plt.show()


def extractDictionaries(graph: nx.Graph) -> None:
    """Extract distributions dictionaries of a nx.Graph objects

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    """
    pixel_dist_school(graph, output=True)
    dictionary = histDistributionLog(graph, logX=False, logY=True, output=True)
    pickleDump(dictionary, "./data/Degreedistribution_Day1.pkl")
    dictionary = histDistributionLog(graph, logX=False, logY=True, output=True)
    pickleDump(dictionary, "./data/Degreedistribution_Day2.pkl")


def school_hist_distribution(graph: nx.Graph, logX=False, logY=True, twoInOne=False, graph2=None, output=False):
    """Degree distributions figure divided into layers
    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    logX : bool
        Denotes whether or not x-axis should be log-scaled
    logY : bool
        Denotes whether or not y-axis should be log-scaled
    twoInOne : bool
        Whether or not to include another day in the plot. Default is False
    output : bool
        Denotes whether or not the dict of frequencies of interactions should be returned. Default is True
    graph2 : nx.Graph
        Another nx.Graph object that describes interactions in the network
    """
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

        pickleDump(graph1, "./data/DegreeDictwhole1.pkl")
        pickleDump(graph2, "./data/DegreeDictwhole2.pkl")
        pickleDump(off_diagonal1, "./data/DegreeDictOffDiag1.pkl")
        pickleDump(off_diagonal2, "./data/DegreeDictOffDiag2.pkl")
        pickleDump(grade_grade1, "./data/DegreeDictgrade1.pkl")
        pickleDump(grade_grade2, "./data/DegreeDictgrade2.pkl")
        pickleDump(class_class1, "./data/DegreeDictclass1.pkl")
        pickleDump(class_class2, "./data/DegreeDictclass2.pkl")

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
        plt.savefig("./fig_master/DegreedistSubgroupsDay1And2.png", bbox_inches="tight", dpi=150)
        plt.show()


def outlierDist(graph: nx.Graph) -> None:
    """Gets the distribution of all interactions a max node has with others

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


def Average(lst: list) -> float:
    """Returns the average of a list

    Parameters
    ----------
    lst : list
        list of interactions
    """
    return sum(lst) / len(lst)


def powerlawCheck(graph: nx.Graph) -> None:
    """Generates R2 of linear regression of nodes in each layer of the model

    Parameters
    ----------
    graph : nx.Graph
        nx.Graph object that contains interactions at a primary school between Person objects.
    """
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


def plotR(graph: nx.Graph, node: int) -> None:
    """Plots R2 of each layer for node in graph and saves the image in R2_img

    Parameters
    ----------
    graph : nx.Graph
        nx.Graph object that contains interactions at a primary school between Person objects.
    node : int
        The ID of the specific node to plot interactions for
    """
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


def plotAllR(graph: nx.Graph) -> None:
    """Plots all R2 of each layer for every node in graph

    Parameters
    ----------
    graph : nx.Graph
        nx.Graph object that contains interactions at a primary school between Person objects.
    """

    for node in graph.nodes():
        plotR(graph, node)


def getClasses(G: nx.Graph):
    """Returs the class modules to be input into the nx.community.modularity funtion

    Parameters
    ----------
    G : nx.Graph
        nx.Graph object that contains interactions at a primary school between Person objects.
    """

    grades = nx.get_node_attributes(G, "klasse")

    d = sorted(grades)

    for node in grades:
        grades = nx.get_node_attributes(G, "klasse")
        classesDict = {}
        for node in grades:
            cl = grades[node]
            if cl not in classesDict:
                classesDict[cl] = {node}
            else:
                classesDict[cl].add(node)
        classesList = []
        for grade in classesDict.keys():
            classesList.append(classesDict[grade])
    return classesList


def modularity(G: nx.Graph):
    """Calculates weighted modularity using the nx.community.modularity funtion

    Parameters
    ----------
    G : nx.Graph
        nx.Graph object that contains interactions at a primary school between Person objects.
    """
    communities = getClasses(G)

    M = community.modularity(G, communities, "weight")

    return M


def general_analysis(graph: nx.Graph) -> None:
    """General network analysis of a graph using NetworkX functions

    Parameters
    ----------
    G : nx.Graph
        nx.Graph object that contains interactions at a primary school between Person objects.
    """
    print("-----------------------------------")
    print(f"# of Nodes: {len(graph.nodes())}")
    print("-----------------------------------")
    print(f"# of edges: {len(graph.edges())}")
    print("-----------------------------------")
    degrees = []
    for deg in graph.degree:
        degrees.append(deg[1])
    print(f"Average degree: {np.mean(degrees)}")
    print("-----------------------------------")
    print(f"Network diameter: {nx.diameter(graph)}")
    print("-----------------------------------")
    aver_short = nx.average_shortest_path_length(graph, weight="weight")
    print(f"Average shortest path : {aver_short}")
    print("-----------------------------------")
    print(f"Average clustering: {nx.average_clustering(graph)}")
    print("-----------------------------------")
    weight_clust = nx.average_clustering(graph, weight="weight")
    print(f"Weighted average clustering: {weight_clust}")
    print("-----------------------------------")
    print(f"Network density: {nx.density(graph)}")
    print("-----------------------------------")
    print(f"Heterogeneity in cytoscape")
    print("-----------------------------------")
    cent = []
    for id, ce in nx.degree_centrality(graph).items():
        cent.append(ce)
    print(f"Average closeness centrality: {np.mean(cent)}")
    print("-----------------------------------")
    betw = []
    for id, bet in nx.betweenness_centrality(graph).items():
        betw.append(bet)
    print(f"Average betweenness centrality: {np.mean(betw)}")
    print("-----------------------------------")
