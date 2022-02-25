from cProfile import label
from cmath import e
import igraph
import numpy as np
import pathpy as pp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from IPython.display import *
from IPython.display import HTML


def create_temp(filename):
    names = ["time", "sourceID", "targetID", "sourceGrade", "targetGrade"]
    days = pd.read_csv(filename, sep="\t", names=names)

    day1 = days[days["time"] < 117240]
    day2 = days[days["time"] >= 117240]

    day1["time"] = day1["time"].apply(lambda x: pd.Timestamp(x, unit="s").hour)
    day2["time"] = day2["time"].apply(lambda x: pd.Timestamp(x, unit="s").hour)

    return day1, day2


def toHour(x):
    return pd.Timestamp(x, unit="s").hour


def plotDailyContacts(filename):
    day1, day2 = create_temp(filename)
    dailyContactDays1 = {}
    dailyContactDays2 = {}

    for i in range(8, day1["time"].max()):
        dailyContactDays1[i] = day1[day1["time"] == i].shape[0]

    for i in range(8, day2["time"].max()):
        dailyContactDays2[i] = day2[day2["time"] == i].shape[0]

    plt.scatter(dailyContactDays1.keys(), dailyContactDays1.values(), color="seagreen", label="Day 1")
    plt.plot(dailyContactDays1.keys(), dailyContactDays1.values(), color="seagreen")
    plt.scatter(dailyContactDays2.keys(), dailyContactDays2.values(), color="coral", label="Day 2")
    plt.plot(dailyContactDays2.keys(), dailyContactDays2.values(), color="coral")
    plt.legend()
    plt.xlabel("Hours since midnight")
    plt.ylabel("Degree")
    plt.savefig("dailyContacts_hourly.png", bbox_inches="tight", dpi=150)
    plt.show()


def hourlyGraph(n, file, day):
    day1, day2 = create_temp(file)
    if day:
        g = day1
    else:
        g = day2
    graph = nx.Graph()
    for index, row in g.iterrows():
        sourceID = int(row["sourceID"])
        targetID = int(row["targetID"])
        time = int(row["time"])
        sourceGrade = row["sourceGrade"]
        targetGrade = row["targetGrade"]

        if time == n:
            found = False
            if (sourceID not in graph.nodes()) and (targetID not in graph.nodes()):
                graph.add_nodes_from([(sourceID, {"grade": sourceGrade})])
                graph.add_nodes_from([(targetID, {"grade": targetGrade})])
                # graph.add_edge(sourceID, targetID, weight=1)
                graph.add_edge(sourceID, targetID, weight=1)

            elif (sourceID in graph.nodes()) and (targetID not in graph.nodes()):
                graph.add_nodes_from([(targetID, {"grade": targetGrade})])
                # graph.add_edge(sourceID, targetID, weight=1)
                graph.add_edge(sourceID, targetID, weight=1)

            elif (targetID in graph.nodes()) and (sourceID not in graph.nodes()):
                graph.add_nodes_from([(sourceID, {"grade": sourceGrade})])
                # graph.add_edge(sourceID, targetID, weight=1)
                graph.add_edge(sourceID, targetID, weight=1)

            else:
                try:
                    graph.edges[(sourceID, targetID)]["weight"] += 1
                except:
                    graph.add_edge(sourceID, targetID, weight=1)
    return graph


def maxHour(file, day):
    day1, day2 = create_temp(file)
    if day:
        g = day1
    else:
        g = day2
    return g["time"].max()


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


def histDistributionLog(graph, logX, logY, output=False, day1=True, wait=False, label=None):
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
    d = toCumulative(sorteddata)

    if day1:
        if label:
            plt.plot(d.keys(), d.values(), "--", label=label)
        else:
            plt.plot(d.keys(), d.values(), label="Day 1", color="seagreen")
    else:
        if label:
            plt.plot(d.keys(), d.values(), color="seagreen", label=label)
        else:
            plt.plot(d.keys(), d.values(), label="Day 2", color="seagreen")

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
        return d.keys(), d.values()

    if not wait:
        plt.show()


def histDistWholeday(file, day, logX=False, logY=True):
    hour = maxHour(file, day)
    print(hour)
    for i in range(8, hour + 1):
        graph = hourlyGraph(i, file, day)
        if i <= 9:
            lab = "0" + str(i) + ":00"
        else:
            lab = str(i) + ":00"
        histDistributionLog(graph, logX, logY, wait=True, label=lab)
    plt.legend()
    plt.savefig("CumulativeDistHourly_Day2.png", bbox_inches="tight", dpi=150)
    plt.show()


file = "Realprimaryschool.csv"
# g = hourlyGraph(n=8, file="Realprimaryschool.csv", day=True)
# histDistributionLog(g, False, True)

histDistWholeday(file, day=False, logX=False, logY=True)
# plotDailyContacts('Realprimaryschool.csv')
