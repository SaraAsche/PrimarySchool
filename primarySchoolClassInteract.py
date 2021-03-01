import itertools
import csv

#Feil å gjøre det slik, teacher blir kun telt som 1 og ingen interaksjoner mellom folk i ulike klasser

dayOne = []
dayTwo = []

def checkIfInteractionExists(day, temp):
    inInteraction = False
    for interaction in day:
            if (temp[3] == interaction[3]) and (temp[4] == interaction[4]):
                inInteraction = True
                if len(interaction)<6:
                    interaction.append(1)
                else:
                    interaction[5] += 1
                break
    if not inInteraction:
        day.append(temp)
                

with open('Realprimaryschool.csv', mode='r') as primarySchoolData:
    
    for line in primarySchoolData:
        temp = list(map(lambda x: x.strip(), line.split("\t")))
        if int(temp[0]) < 117240:
            if (temp[3] in list(itertools.chain(*dayOne))):
                checkIfInteractionExists(dayOne, temp)
            else:
                dayOne.append(temp)
        else:
            if (temp[3] in list(itertools.chain(*dayTwo))):
                checkIfInteractionExists(dayTwo, temp)
            else:
                dayTwo.append(temp)
    

with open('dayOnePrimaryClass.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for interaction in dayOne:
        writer.writerow(interaction)

with open('dayTwoPrimaryClass.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for interaction in dayTwo:
        writer.writerow(interaction)

