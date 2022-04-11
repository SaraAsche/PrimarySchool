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
from itertools import groupby


day1 = nx.Graph()  # day1
day2 = nx.Graph()  # day2

with open("dayOneNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        day1.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        day1.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        day1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        day1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

with open("dayTwoNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        day2.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        day2.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        # graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        day2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        day2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

color_map = []
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


def generate_heatmap(graph):
    a_list = list(graph.nodes)
    a_list.sort()
    A = nx.adjacency_matrix(graph, nodelist=a_list)

    A_M = A.todense()

    sns.heatmap(A_M)
    # plt.savefig(f'./fig_master/Heatmap_day2.png', bbox_inches='tight', dpi=500)
    plt.show()


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


def plot_graph(graph):
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

    nx.draw_networkx_edges(Gcc, pos, alpha=0.4, ax=ax)
    plt.legend(scatterpoints=1, frameon=False)  # ["1", "2", "3", "4", "5"],
    ax.axis("off")
    plt.savefig("./fig_master/Day1_network.png", transparent=True, dpi=500)
    plt.show()


def degree_dist(G):
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
    plt.savefig("./fig_master/Erdős-Rényi.png", transparent=True, dpi=500)
    plt.show()


def generate_random_networks(ER=True, WS=False, BA=False):
    if ER:
        return nx.erdos_renyi_graph(236, 0.21)
    elif WS:
        return nx.watts_strogatz_graph(236, 50, 0.4)
    elif BA:
        return nx.barabasi_albert_graph(n=236, m=28)


plot_graph(day1)
