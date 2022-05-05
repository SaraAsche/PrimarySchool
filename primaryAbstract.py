"""primaryAbstract.py

Not used in thesis, an attempt to produce an abstract primary school model 

Author: Sara Johanne Asche
Date: 05.05.2022
File: primaryAbstract.py
"""

from pprint import pprint

attrs = {}
layers = {"students": [], "class": {"cliques": [], "open": True}, "grade": {"cliques": [], "open": True}, "teacher": []}


def generateSchool(
    classesperGrade, numberOfGrades, numTeachersPerClass
):  # input: number of students at school and grades. Primary school == 7, junior high school == 3.
    klasse = 0
    for grade in range(1, numberOfGrades + 1):
        tempstudents = []
        tempClasses = []
        count = 0 if len(layers["class"]["cliques"]) == 0 else layers["teacher"][-1]
        for student in range(count, 23 * classesperGrade * grade + 1):
            if student % 23 == 0 and student:
                tempClass = tempstudents.copy()
                for teacher in range(numTeachersPerClass):
                    t = tempClass[-1] + 1
                    layers["teacher"].append(t)
                    tempClass.append(t)
                    fillAttrs(teacher, layers, False, [grade, klasse])
                layers["class"]["cliques"].append({"nodes": tempClass.copy(), "open": True})
                klasse += 1
                tempClasses.append(tempClass)
                tempstudents = []
                if len(layers["teacher"]) / 2 == numberOfGrades * classesperGrade:
                    break

            tempstudents.append(student + len(layers["teacher"]))
            layers["students"].append(student + len(layers["teacher"]))
            fillAttrs(student, layers, True, [grade - 1, klasse])

        layers["grade"]["cliques"].append({"nodes": [i for klasse in tempClasses.copy() for i in klasse], "open": True})


def fillAttrs(node, layers, child, gradeAndClass):
    if child:
        attrs[node] = {}
        attrs[node]["age"] = 10
        attrs[node]["decade"] = 10
        attrs[node]["ageGroup"] = "A1"
        attrs[node]["cliques"] = []
        attrs[node]["state"] = "S"
        attrs[node]["quarantine"] = False
        attrs[node]["sick"] = False
        attrs[node]["inNursing"] = False
        attrs[node]["present"] = {}
        attrs[node]["cliques"].append(["grade", gradeAndClass[0]])
        attrs[node]["cliques"].append(["class", gradeAndClass[1]])
        attrs[node]["cliques"].append(["students", 1])
    else:
        attrs[node] = {}
        attrs[node]["age"] = 40
        attrs[node]["decade"] = 40
        attrs[node]["ageGroup"] = "A1"
        attrs[node]["cliques"] = []
        attrs[node]["state"] = "S"
        attrs[node]["quarantine"] = False
        attrs[node]["sick"] = False
        attrs[node]["inNursing"] = False
        attrs[node]["present"] = {}

        attrs[node]["cliques"].append(["grade", gradeAndClass[0]])
        attrs[node]["cliques"].append(["class", gradeAndClass[1]])
        attrs[node]["cliques"].append(["teacher", 1])


generateSchool(3, 3, 2)
pprint(layers)
