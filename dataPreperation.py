"""dataPreparation.py

A file for the data preparation that was done for the primary school interaction dataset.
Loops through all 20 seconds interactions and returns an accumulated weighted network.

Author: Sara Johanne Asche
Date: 05.05.2022
File: dataPreperation.py
"""

import itertools
import csv
import pandas as pd


def checkIfInteractionExists(day, temp):
    """Checks if interaction between two individuals already exists. Adds to the weight between individuals
    if it already exists. If not, a new interaction between two individual is made with weight 1.

    Parameters
    ----------
    day : list
        List for day one or day two containing interactions
    temp : list
        List of the 20 second interaction that is checked between two individuals
    """
    inInteraction = False
    for interaction in day:
        if (temp[1] in interaction) and (temp[2] in interaction):
            inInteraction = True
            if len(interaction) < 6:
                interaction.append(1)
            else:
                interaction[5] += 1
            break
    if not inInteraction:
        temp.append(1)
        day.append(temp)


def add_to_csv():
    """Accumulates the 20 seconds interactions to a new csv file with weight between two individuals being how many times they have interacted"""
    dayOne = []
    dayTwo = []

    # Realprimaryschool.csv contains both day 1 and day 2
    with open("./data/Realprimaryschool.csv", mode="r") as primarySchoolData:
        for line in primarySchoolData:
            temp = list(map(lambda x: x.strip(), line.split("\t")))
            # First day is < 117240 seconds
            if int(temp[0]) < 117240:
                if temp[1] in list(itertools.chain(*dayOne)):
                    checkIfInteractionExists(dayOne, temp)
                else:
                    temp.append(1)
                    dayOne.append(temp)
            else:
                if temp[1] in list(itertools.chain(*dayTwo)):
                    checkIfInteractionExists(dayTwo, temp)
                else:
                    temp.append(1)
                    dayTwo.append(temp)
    # Saves accumulated day one
    with open("./data/dataPreperation/dayOnePrimary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for interaction in dayOne:
            writer.writerow(interaction)

    # Saves accumulated day two
    with open("./data/dataPreperation/dayTwoPrimary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for interaction in dayTwo:
            writer.writerow(interaction)


def generate_sorted_indexes():
    """Generates sorted indexes for each individual with ID increasing with grade"""
    dayOne = pd.read_csv("./data/dataPreperation/dayOnePrimary.csv", header=None)
    dayTwo = pd.read_csv("./data/dataPreperation/dayTwoPrimary.csv", header=None)
    metadata = pd.read_csv("./data/metadata_primaryschool.txt", delimiter="\t", header=None)
    dayOne.columns = ["Time", "SourceID", "TargetID", "SourceClass", "TargetClass", "NumberOfInteractions"]
    dayTwo.columns = ["Time", "SourceID", "TargetID", "SourceClass", "TargetClass", "NumberOfInteractions"]

    metadata.columns = ["ID", "Class", "Gender"]
    metadata.sort_values(["Class", "ID"], inplace=True)
    metadata.reset_index(drop=True, inplace=True)

    fixId(dayOne, metadata)
    fixId(dayTwo, metadata)

    metadata.to_csv("./data/dataPreperation/newMetadata.csv", header=None)

    dayOne.to_csv("./data/dataPreperation/dayOneNewIndex.csv", header=None, index=False)
    dayTwo.to_csv("./data/dataPreperation/dayTwoNewIndex.csv", header=None, index=False)


def fixId(df, metadata):
    """Sets the ID of df to the same as in metadata

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe containing all accumulated interactions for a day
    metadata : pd.Dataframe
        Dataframe with updated ID for all individuals in the school
    """
    df["OldSourceIndex"] = pd.Series([None] * len(df), index=df.index)
    df["OldTargetIndex"] = pd.Series([None] * len(df), index=df.index)

    for i, row in df.iterrows():
        oldSrc = row["SourceID"]
        oldTrgt = row["TargetID"]

        df.iloc[i, 1] = metadata[metadata["ID"] == df.iloc[i, 1]].index[0]
        df.iloc[i, 2] = metadata[metadata["ID"] == df.iloc[i, 2]].index[0]

        df.iloc[i, -2] = oldSrc  # metadata[metadata["ID"] == dayOne.iloc[i, 1]].index[0]
        df.iloc[i, -1] = oldTrgt  # metadata[metadata["ID"] == dayOne.iloc[i, 2]].index[0]


def maxmin():
    """Calculates the maximum and minimum value of the old IDs"""
    metadata = pd.read_csv("./data/metadata_primaryschool.txt", delimiter="\t", header=None)
    metadata.columns = ["ID", "Class", "Gender"]

    column = metadata["ID"]
    max_value = column.max()
    min_value = column.min()
    print(f"max value is: {max_value}, and min value is: {min_value}")
