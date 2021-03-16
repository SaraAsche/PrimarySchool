import pandas as pandas
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
import numpy as np

dayOne = pandas.read_csv('dayOnePrimary.csv', header=None)
dayTwo = pandas.read_csv('dayTwoPrimary.csv', header = None)
metadata = pandas.read_csv('metadata_primaryschool.txt', delimiter='\t', header = None)
dayOne.columns = ['Time', 'SourceID', 'TargetID', 'SourceClass', 'TargetClass', 'NumberOfInteractions']
dayTwo.columns = ['Time', 'SourceID', 'TargetID', 'SourceClass', 'TargetClass', 'NumberOfInteractions']

metadata.columns = ['ID', 'Class', 'Gender']
metadata.sort_values(['Class', 'ID'], inplace=True)
metadata.reset_index(drop=True, inplace=True)



for i, row in dayOne.iterrows():
    dayOne.iloc[i, 1] = metadata[metadata['ID'] == dayOne.iloc[i, 1]].index[0]
    dayOne.iloc[i, 2] = metadata[metadata['ID'] == dayOne.iloc[i, 2]].index[0]

for i, row in dayTwo.iterrows():
    dayTwo.iloc[i, 1] = metadata[metadata['ID'] == dayTwo.iloc[i, 1]].index[0]
    dayTwo.iloc[i, 2] = metadata[metadata['ID'] == dayTwo.iloc[i, 2]].index[0]

metadata.to_csv('newMetadata.csv', header = None)

dayOne.to_csv('dayOneNewIndex.csv', header = None, index = False)
dayTwo.to_csv('dayTwoNewIndex.csv', header = None, index = False)
