from curses import meta
import itertools
import csv
import pandas as pd


def checkIfInteractionExists(day, temp):
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
    dayOne = []
    dayTwo = []

    with open("Realprimaryschool.csv", mode="r") as primarySchoolData:
        for line in primarySchoolData:
            temp = list(map(lambda x: x.strip(), line.split("\t")))
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

    with open("./dataPreperation/dayOnePrimary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for interaction in dayOne:
            writer.writerow(interaction)

    with open("./dataPreperation/dayTwoPrimary.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for interaction in dayTwo:
            writer.writerow(interaction)


def generate_sorted_indexes():
    dayOne = pd.read_csv("./dataPreperation/dayOnePrimary.csv", header=None)
    dayTwo = pd.read_csv("./dataPreperation/dayTwoPrimary.csv", header=None)
    metadata = pd.read_csv("metadata_primaryschool.txt", delimiter="\t", header=None)
    dayOne.columns = ["Time", "SourceID", "TargetID", "SourceClass", "TargetClass", "NumberOfInteractions"]
    dayTwo.columns = ["Time", "SourceID", "TargetID", "SourceClass", "TargetClass", "NumberOfInteractions"]

    metadata.columns = ["ID", "Class", "Gender"]
    metadata.sort_values(["Class", "ID"], inplace=True)
    metadata.reset_index(drop=True, inplace=True)

    fixId(dayOne, metadata)
    fixId(dayTwo, metadata)

    metadata.to_csv("./dataPreperation/newMetadata.csv", header=None)

    dayOne.to_csv("./dataPreperation/dayOneNewIndex.csv", header=None, index=False)
    dayTwo.to_csv("./dataPreperation/dayTwoNewIndex.csv", header=None, index=False)


def fixId(df, metadata):
    df["OldSourceIndex"] = pd.Series([None] * len(df), index=df.index)
    df["OldTargetIndex"] = pd.Series([None] * len(df), index=df.index)

    for i, row in df.iterrows():
        oldSrc = row["SourceID"]
        oldTrgt = row["TargetID"]

        df.iloc[i, 1] = metadata[metadata["ID"] == df.iloc[i, 1]].index[0]
        df.iloc[i, 2] = metadata[metadata["ID"] == df.iloc[i, 2]].index[0]

        df.iloc[i, -2] = oldSrc  # metadata[metadata["ID"] == dayOne.iloc[i, 1]].index[0]
        df.iloc[i, -1] = oldTrgt  # metadata[metadata["ID"] == dayOne.iloc[i, 2]].index[0]


def main():
    add_to_csv()
    generate_sorted_indexes()


def maxmin():
    metadata = pd.read_csv("metadata_primaryschool.txt", delimiter="\t", header=None)
    metadata.columns = ["ID", "Class", "Gender"]

    column = metadata["ID"]
    max_value = column.max()
    min_value = column.min()
    print(f"max value is: {max_value}, and min value is: {min_value}")


maxmin()
# generate_sorted_indexes()
