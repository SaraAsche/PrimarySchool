import itertools
import csv
from datetime import timedelta


dayOne = []
dayTwo = []

def checkIfInteractionExists(day, temp):
    inInteraction = False
    for interaction in day:
            if (temp[1] in interaction) and (temp[2] in interaction):
                inInteraction = True
                if len(interaction)<6:
                    interaction.append(1)
                else:
                    interaction[5] += 1
                break
    if not inInteraction:
        temp.append(1)
        day.append(temp)

with open('Realprimaryschool.csv', mode='r') as primarySchoolData:
    
    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split("\t")))
        time = timedelta(seconds=int(temp[0]))
        
        lunchbreakStart = timedelta(hours=12)
        lunchbreakEnd = timedelta(hours=14)
        if int(temp[0]) < 117240:
            if (time>=lunchbreakStart)&(time<=lunchbreakEnd):
                if (temp[1] in list(itertools.chain(*dayOne))):
                    checkIfInteractionExists(dayOne, temp)
                else:
                    temp.append(1)
                    dayOne.append(temp)    
        else:
            if (time>=timedelta(hours=36))&(time<=timedelta(hours=38)):
                if (temp[1] in list(itertools.chain(*dayTwo))):
                    checkIfInteractionExists(dayTwo, temp)
                else:
                    temp.append(1)
                    dayTwo.append(temp)


with open('dayOnePrimaryLunch.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for interaction in dayOne:
        writer.writerow(interaction)

with open('dayTwoPrimaryLunch.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for interaction in dayTwo:
        writer.writerow(interaction)

