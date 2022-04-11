import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from networkx.algorithms import community


def loadNetwork(name):
    return pickle.load(open(name, "rb"))


withoutGrade_day1 = loadNetwork("withoutGrade_day1")
withoutGrade_day2 = loadNetwork("withoutGrade_day2")

onlyGrade_day1 = loadNetwork("onlyGrade_day1")
onlyGrade_day2 = loadNetwork("onlyGrade_day2")

withoutGrade_day1_Pearson = nx.degree_pearson_correlation_coefficient(withoutGrade_day1)


def plotDegreeDegreeConnection(graph, weight):
    xdata = []
    ydata = []

    if weight:
        for i, j in graph.edges():
            xdata.append(graph.degree(i, weight="weight"))
            ydata.append(graph.degree(j, weight="weight"))
            xdata.append(graph.degree(j, weight="weight"))
            ydata.append(graph.degree(i, weight="weight"))
    else:
        for i, j in withoutGrade_day1.edges():
            xdata.append(graph.degree(i))
            ydata.append(graph.degree(j))
            xdata.append(graph.degree(j))
            ydata.append(graph.degree(i))

    """
    heatmap, xedges, yedges = np.histogram2d(xdata, ydata, bins=25)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin ='lower')
    plt.show()
    """

    h = sns.jointplot(x=xdata, y=ydata, kind="hist", cbar=True)

    h.set_axis_labels("Node i degree <k>", "Node j degree <k>", fontsize=10)
    plt.tight_layout()
    plt.savefig(
        "./fig_master/Assortativity.png",
        transparent=True,
        dpi=500,
    )
    plt.show()

    # plt.plot(xdata, ydata, "o", alpha=0.05)
    # plt.ylabel("Edge with degree <k>")
    # plt.xlabel("Edge with degree <k>")
    # plt.title("Diagonal representation of edges connecting two nodes")
    plt.show()


plotDegreeDegreeConnection(onlyGrade_day2, True)

print(withoutGrade_day1_Pearson)  # Pearson correlation is positive
