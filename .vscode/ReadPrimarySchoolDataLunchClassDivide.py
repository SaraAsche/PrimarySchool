import itertools
import csv
from datetime import timedelta
import matplotlib.pyplot as plt

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


def readPrimaryschool(klasse, day):
    with open('Realprimaryschool.csv', mode='r') as primarySchoolData:
        newLunch = []
        separation_points=[]

        if (day == 1):
            lunchbreak_start = timedelta(hours=12)
            lunchbreak_end = timedelta(hours=14)
        else:
            lunchbreak_start = timedelta(hours=36)
            lunchbreak_end = timedelta(hours=38)


        timespan=lunchbreak_start; #Nullpunktet

        for line in primarySchoolData:
            temp = list(map(lambda x: x.strip(), line.split("\t")))
            time=timedelta(seconds=int(temp[0])) #Antall sekunder fra midnatt

            if (time>=lunchbreak_start and time<=lunchbreak_end and (temp[3][0]==klasse or temp[4][0]==klasse)):
                if((time-timespan)>=timedelta(minutes=10)):
                    newLunch.append(separation_points)
                    separation_points=[]
                    timespan=time

                separation_points.append(temp)
        
    return newLunch
            
            
            
def readPrimaryschool2(klasse, day):
    with open('Realprimaryschool.csv', mode='r') as primarySchoolData:
        newLunch = []
        separation_points=[]

        if (day == 1):
            dayOne=True
        else:
            dayOne=False
        
        if day==1:
            timespan=timedelta(seconds=31220) #Nullpunktet
            dayOne=True
        if day==2:
            timespan=timedelta(seconds=117240)
            dayOne=False

        for line in primarySchoolData:
            temp = list(map(lambda x: x.strip(), line.split("\t")))
            time=timedelta(seconds=int(temp[0])) #Antall sekunder fra midnatt

            if (dayOne and int(temp[0])<117240 and (temp[3][0]==klasse or temp[4][0]==klasse)):
                if((time-timespan)>=timedelta(minutes=10)):
                    newLunch.append(separation_points)
                    separation_points=[]
                    timespan=time

                separation_points.append(temp)

            if (not dayOne and int(temp[0])>=117240 and (temp[3][0]==klasse or temp[4][0]==klasse)):
                if((time-timespan)>=timedelta(minutes=10)):
                    newLunch.append(separation_points)
                    separation_points=[]
                    timespan=time

                separation_points.append(temp)

    print(len(newLunch))
    return newLunch


def checkInteractLunch(listOfClass, day):
    temporary = []

    tenmin= []

    finalInt = []

    for tenMinutes in listOfClass:
        temporary=[]
        for interaction in tenMinutes:
            if (interaction[1] in list(itertools.chain(*temporary))):
                checkIfInteractionExists(temporary, interaction)
            else:
                interaction.append(1)
                temporary.append(interaction)
        finalInt.append(temporary) 
    
    return finalInt
        

checkInteractLunch(readPrimaryschool('1', 2), 2)

'''
with open('dayOnePrimaryLunch10minClass5.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for interaction in checkInteractLunch(readPrimaryschool('1', 1), 1):
        for line in interaction:
            writer.writerow(line)

with open('dayTwoPrimaryLunch10minClass5.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for interaction in checkInteractLunch(readPrimaryschool('1', 2), 2):
        for line in interaction:
            writer.writerow(line)


'''

alle_klasse_degrees = []
for i in range(1, 6):
    klasse_degrees = []
    for interaction in checkInteractLunch(readPrimaryschool2(str(i), 2), 2):
        klasse_degrees.append(sum(map(lambda x: x[-1], interaction)))
    #print(len(klasse_degrees))
    alle_klasse_degrees.append(klasse_degrees)

time=520
Y=[]
for i in alle_klasse_degrees[0]:
    Y.append(time)
    time= time+10



colors = {
    0: 'blue',
    1: 'magenta',
    2: 'gold',
    3: 'red',
    4: 'brown'
}

print(alle_klasse_degrees)

for i, klasse_degrees in enumerate(alle_klasse_degrees):
    #print('Y='+str(len(Y)))
    #print('X='+str(len(klasse_degrees)))
    if i==3:
        continue
    
    plt.scatter(Y, klasse_degrees, c=colors[i], label=f'{i+1}. class', alpha=0.8)
    
plt.ylabel('Number of interactions')
plt.xlabel('Minutes since midnight')
plt.legend()
plt.show()

for i, klasse_degrees in enumerate(alle_klasse_degrees):
    if i==3:
        continue
    
    plt.plot(Y, klasse_degrees, c=colors[i], label=f'{i+1}. class', alpha=0.8, linestyle='dashed')
    
plt.legend()
plt.ylabel('Number of interactions')
plt.xlabel('Minutes since midnight')
plt.show()

# print(alle_klasse_degrees)



# klasse_degrees = [sum(list(map(lambda x: x[-1], checkInteractLunch(readPrimaryschool(str(i), 1), 1)))) for i in range(1, 6)]
# print(len(checkInteractLunch(readPrimaryschool(str(1), 1), 1)))
# print(list(map(lambda x: x[-1], checkInteractLunch(readPrimaryschool(str(1), 1), 1))))