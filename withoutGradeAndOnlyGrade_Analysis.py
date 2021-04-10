import pickle
import networkx as nx

from networkx.algorithms import community

def loadNetwork(name):
    return pickle.load(open(name, 'rb'))

withoutGrade_day1 = loadNetwork('withoutGrade_day1')
withoutGrade_day2 = loadNetwork('withoutGrade_day2')

onlyGrade_day1 = loadNetwork('onlyGrade_day1')
onlyGrade_day2 = loadNetwork('onlyGrade_day2')

communities = community.greedy_modularity_communities(withoutGrade_day1)

print(communities)