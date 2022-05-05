"""Create_images.py

A file for generating images for analysis of primary school data. 
Provides generation of heatmaps, degree distributions, interaction distributions,
the graph with each node coloured by their grade. It also provides the functionality 
for generating random networks and subgraphs of the original network. In addition it shows
the plotting done for assortativity according to grade as well as investigating if there is a
relationship between genders of the pupils and who they chose to interact with. 


Author: Sara Johanne Asche
Date: 05.05.2022
File: create_images.py
"""


from pstats import Stats
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d


day1 = nx.Graph()  # day1
day2 = nx.Graph()  # day2

# Loading day 1
with open("./data/dayOneNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        day1.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        day1.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        day1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        day1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

# Loading day 2
with open("./data/dayTwoNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        day2.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        day2.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        # graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        day2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        day2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

# Loading metadata and adding it to day one and two
with open("./data/dataPreperation/newMetadata.csv", mode="r") as metadat:
    for line in metadat:

        temp = list(map(lambda x: x.strip(), line.split(",")))

        for node in day1:

            if str(node) == temp[0]:
                day1.nodes[node]["sex"] = temp[3]

        for node in day2:

            if str(node) == temp[0]:
                day2.nodes[node]["sex"] = temp[3]


# Creating an overal color_map and weights for plotting of graph
color_map = []
weights = [None for _ in range(len(day1.edges))]
grades = nx.get_node_attributes(day1, "klasse")

for node in day1:
    if grades[node][0] == str(1):
        color_map.append("rosybrown")
    elif grades[node][0] == str(2):
        color_map.append("sienna")
    elif grades[node][0] == str(3):
        color_map.append("tan")
    elif grades[node][0] == str(4):
        color_map.append("darkgoldenrod")
    elif grades[node][0] == str(5):
        color_map.append("olivedrab")
    else:
        color_map.append("slategrey")

maximum_count = max(list(map(lambda x: x[-1]["weight"], day1.edges(data=True))))
for i, e in enumerate(day1.edges(data=True)):
    weights[i] = (0, 0, 0, e[-1]["weight"] / (maximum_count - 300))

#
def generate_heatmap(graph, output=False):
    """Generates a heatmap from a nx.Graph object

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    output : bool
        whether or not to return the adjacency matrix. Default is to not return the matrix
    """

    a_list = list(graph.nodes)
    a_list.sort()
    A = nx.adjacency_matrix(graph, nodelist=a_list)
    if output:
        return A
    A_M = A.todense()

    sns.heatmap(A_M)  #
    plt.savefig(f"./fig_master/heatmap_class.png", bbox_inches="tight", dpi=500)
    plt.show()


def toCumulative(l):
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


def plot_graph(graph):
    """Plots a nx.Graph object with nodes coloured by grades

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    """
    sizes = [100 for _ in range(len(graph.nodes))]
    color_map2 = {1: "rosybrown", 2: "sienna", 3: "tan", 4: "darkgoldenrod", 5: "olivedrab"}
    fig, ax = plt.subplots()
    Gcc = graph
    pos = nx.spring_layout(Gcc, seed=10396953)
    for i in range(1, 6):
        nodes = [x[0] for x in list(filter(lambda x: x[1]["klasse"][0] == str(i), Gcc.nodes(data=True)))]
        print(len(nodes))
        nx.draw_networkx_nodes(Gcc, pos, nodelist=nodes, node_size=20, node_color=color_map2[i], label=str(i), ax=ax)
        print(color_map[i])

    nx.draw_networkx_nodes(
        Gcc,
        pos,
        nodelist=[x[0] for x in list(filter(lambda x: x[1]["klasse"] == "Teachers", Gcc.nodes(data=True)))],
        node_size=20,
        node_color="slategray",
        label="Teachers",
        ax=ax,
    )

    nx.draw_networkx_edges(
        Gcc,
        pos,
        ax=ax,
        edge_color=weights,
    )  # alpha=0.8
    plt.legend(scatterpoints=1, frameon=False)  # ["1", "2", "3", "4", "5"],
    ax.axis("off")
    plt.savefig("./fig_master/Day1_network.png", transparent=True, dpi=500)
    plt.show()


def degree_dist(G):
    """Plots graph, degree dist and histogram of nx.Graph object

    Parameters
    ----------
    G : nx.Graph
        A nx.Graph object that describes interactions in the network
    """
    degree_sequence = sorted([d for n, d in G.degree(weight="weight")], reverse=False)

    fig = plt.figure("Degree of a random graph", figsize=(8, 8))

    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])

    Gcc = G
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20, node_color=color_map)  #
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("Erdős-Rényi network")
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, 2:])

    degs = {}
    for n in G.nodes():
        deg = G.degree(n, weight="weight")
        degs[n] = deg

    items = sorted(degs.items())

    data = []
    for line in items:
        data.append(line[1])

    sorteddata = np.sort(data)
    print(sorteddata)
    d = toCumulative(sorteddata)

    ax1.plot(d.keys(), d.values(), color="seagreen")

    ax1.set_title("Cumulative degree distribution P(X > x)")
    ax1.set_ylabel("Frequency")
    ax1.set_xlabel("Degree")

    ax2 = fig.add_subplot(axgrid[3:, :2])

    ax2.set_title("Histogram of degree distribution")
    ax2.hist(data, bins=20, color="seagreen", ec="black")  # col = 'skyblue for day2, mediumseagreen for day1
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Frequency")

    fig.tight_layout()
    # plt.savefig("./fig_master/Erdős-Rényi.png", transparent=True, dpi=500)
    plt.show()


def generate_random_networks(ER=True, WS=False, BA=False):
    """Generates random networks with set parameters

    Parameters
    ----------
    ER : bool
        Whether or not to generate Erdoz-Renyi network. Default is True
    WS : bool
        Whether or not to generate Watts-strogatz network. Default is False
    BA : bool
        Whether or not to generate Barabási-Albert network. Default is False
    """
    if ER:
        return nx.erdos_renyi_graph(236, 0.21)
    elif WS:
        return nx.watts_strogatz_graph(236, 50, 0.4)
    elif BA:
        return nx.barabasi_albert_graph(n=236, m=28)


def plot_deg(G, wait=True, label=None, col="rosybrown"):
    """Generates the degree distribution of a nx.Graph object

    Parameters
    ----------
    G : nx.Graph
        A nx.Graph object that describes interactions in the network
    """
    degs = {}
    for n in G.nodes():
        deg = G.degree(n, weight="weight")
        degs[n] = deg

    items = sorted(degs.items())

    data = []
    for line in items:
        data.append(line[1])

    sorteddata = np.sort(data)
    print(sorteddata)
    d = toCumulative(sorteddata)

    xdat = np.array(list(d.keys()))
    ydat = np.array(list(d.values()))

    plt.rcParams.update({"font.size": 16})
    ysmoothed = gaussian_filter1d(ydat, sigma=2)
    xsmoothed = gaussian_filter1d(xdat, sigma=2)
    plt.plot(xsmoothed, ysmoothed, label=label, linewidth=3, color=col)
    plt.yscale("log")

    plt.xlabel("Degree")
    plt.ylabel("log P(X<x)")
    if wait:
        return
    else:
        plt.tight_layout()
        plt.legend()
        plt.savefig("./fig_master/degree_layers_with_legend.png", transparent=True, dpi=500)
        plt.show()


def plotDegreeDegreeConnection(graph, weight):
    """Plots assortativity between connected nodes with regard to degree

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    weight : bool
        Denotes whether or not weight of interaction is accounted for.
    """
    xdata = []
    ydata = []

    if weight:
        for i, j in graph.edges():
            xdata.append(graph.degree(i, weight="weight"))
            ydata.append(graph.degree(j, weight="weight"))
            xdata.append(graph.degree(j, weight="weight"))
            ydata.append(graph.degree(i, weight="weight"))
    else:
        for i, j in graph.edges():
            xdata.append(graph.degree(i))
            ydata.append(graph.degree(j))
            xdata.append(graph.degree(j))
            ydata.append(graph.degree(i))

    sns.jointplot(x=xdata, y=ydata, kind="hist", cbar=True, color="darkslategrey")  # hist,  cbar=True

    plt.tight_layout()

    plt.savefig("./fig_master/Assortativity2.png", transparent=True, dpi=500)

    plt.show()

    sns.regplot(x=xdata, y=ydata, scatter=False, fit_reg=True, color="darkslategrey")
    plt.tight_layout()
    plt.ylim(top=1400)
    plt.savefig("./fig_master/reg_assortativity2.png", transparent=True, dpi=500)
    plt.show()


def createSubGraphWithout(graph, grade, klasse):
    """Generates a subgraph of a nx.Graph object

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    grade : bool
        Denotes whether or not grade interactions should be included
    klasse : bool
        Denotes whether or not grade interactions should be included
    """
    # Function generate network without all interactions within the same grade. If klasse = true: without the class interactions
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


def pixelDist(graph, logY, logX, axis=None, output=False, label=None, col="rosybrown", wait=True):
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
    label : str
        The label of the graph plotted
    col : str
        matplotlib colour for plot. Default is "rosybrown"
    wait : bool
        Whether or not to plot or wait for more curves to be drawn on same figure before plotting. Default is True
    """
    A = generate_heatmap(graph, output=True)
    length = len(graph.nodes())
    weights = A[np.triu_indices(length, k=1)].tolist()[0]

    data = sorted(weights)

    sorteddata = np.sort(data)
    d = toCumulative(sorteddata)

    if output:
        return d

    else:
        xdat = np.array(list(d.keys()))
        ydat = np.array(list(d.values()))

        plt.rcParams.update({"font.size": 16})
        ysmoothed = gaussian_filter1d(ydat, sigma=2)
        xsmoothed = gaussian_filter1d(xdat, sigma=2)
        plt.plot(xsmoothed, ysmoothed, label=label, linewidth=3, color=col)

        if not wait:
            plt.legend()
            plt.tight_layout(pad=2)
            plt.yscale("log")
            plt.ylabel("Cumulative log(Frequency)")

            plt.xscale("log")
            plt.xlabel("log(Weight)")
            plt.savefig("./fig_master/pixel_dist.png", transparent=True, dpi=500)
            plt.show()


def degree_vs_sex(graph):
    """Investigates the effect of degree vs sex of the individuals for a nx.Graph object

    Parameters
    ----------
    graph : nx.Graph
        A nx.Graph object that describes interactions in the network
    """
    deg = []
    sex = []
    l = []
    for individual in graph.nodes():
        if graph.nodes[individual]["sex"] == "Unknown":
            continue
        else:
            deg.append(graph.degree(individual))
            sex.append(graph.nodes[individual]["sex"])
            l.append((graph.degree(individual), graph.nodes[individual]["sex"]))

    U1, p = Stats.mannwhitneyu(
        list(map(lambda x: x[0], filter(lambda x: x[1] == "M", l))),
        list(map(lambda x: x[0], filter(lambda x: x[1] == "F", l))),
        method="exact",
    )
    print(U1)
    print(p)
    print(len(list(map(lambda x: x[0], filter(lambda x: x[1] == "M", l)))))
    print(len(list(map(lambda x: x[0], filter(lambda x: x[1] == "F", l)))))

    return U1, p


def pixel__dist_layers():
    """Displays the pixel/interaction distribution for all layers in the model"""
    off_diagonal = createSubGraphWithout(day1, True, False)
    grade_grade = createSubGraphWithoutGraph(day1, False, True)
    class_class = createSubGraphWithoutGraph(day1, True, False)

    pixelDist(off_diagonal, logX=True, logY=True, label="Off-diagonal", col="cadetblue")
    pixelDist(grade_grade, logX=True, logY=True, label="Grade", col="darkkhaki")
    pixelDist(day1, logX=True, logY=True, label="Whole", col="silver")
    pixelDist(class_class, logX=True, logY=True, label="Class", wait=False)
