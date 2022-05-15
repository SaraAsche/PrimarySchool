import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

graph1 = nx.Graph()  # day1
graph2 = nx.Graph()  # day2

with open("./data/dayOneNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph1.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph1.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        graph1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph1.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))

with open("./data/dayTwoNewIndex.csv", mode="r") as primarySchoolData:

    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split(",")))
        # weightln = 1+np.log(int(temp[5]))
        graph2.add_nodes_from([(int(temp[1]), {"klasse": temp[3]})])
        graph2.add_nodes_from([(int(temp[2]), {"klasse": temp[4]})])
        # graph.add_edge(int(temp[1]), int(temp[2]), weight = int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))
        graph2.add_edge(int(temp[1]), int(temp[2]), weight=int(temp[5]))


def general_analysis(graph):
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


def histDistributionLog(graph, logX, logY, wait=False, label=None, col="grey"):
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n)  # , weight="weight"
        degs[n] = deg
        # degs[n] = 1-np.log(deg)

    items = sorted(degs.items())

    data = []
    for line in items:
        data.append(line[1])

    sorteddata = np.sort(data)
    print(sorteddata)
    d = toCumulative(sorteddata)

    if label:
        plt.scatter(d.keys(), d.values(), label=label, alpha=0.5, s=20, color=col)
        if logY:
            plt.yscale("log")
            plt.ylabel("Cumulative log frequency", size=13)
        else:
            plt.yscale("linear")
            plt.ylabel("Cumulative frequency", size=13)

        if logX:
            plt.xscale("log")
            plt.xlabel("log Degree", size=13)
        else:
            plt.xscale("linear")
            plt.xlabel("Degree", size=13)

    if not wait:
        plt.legend()
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.tick_params("both", length=10, width=1, which="major")
        plt.tick_params("both", length=5, width=0.5, which="minor")
        plt.savefig("random_networks.png", bbox_inches="tight", dpi=150)
        plt.show()


def main():
    ER = nx.erdos_renyi_graph(237, 0.21)
    WS = nx.watts_strogatz_graph(237, 50, 0.4)
    BA = nx.barabasi_albert_graph(237, 28)

    # histDistributionLog(ER, True, True, wait=True, label="ER", col="darkgoldenrod")
    # histDistributionLog(WS, True, True, wait=True, label="WS", col="olivedrab")
    # histDistributionLog(BA, True, True, wait=True, label="BA", col="tan")
    # histDistributionLog(graph1, True, True, wait=True, label="Day 1", col="rosybrown")
    # histDistributionLog(graph2, True, True, wait=False, label="Day 2", col="cadetblue")

    print("day1")
    general_analysis(graph1)
    print("day2")
    general_analysis(graph2)
    print("ER")
    general_analysis(ER)
    print("WS")
    general_analysis(WS)
    print("BA")
    general_analysis(BA)

