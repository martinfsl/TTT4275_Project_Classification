import numpy as np
import matplotlib.pyplot as plt

# Load the data for the Iris Setosa class
# The data lines are stores in the order: sepal length, sepal width, petal length, petal width - All in cm
with open("Iris_TTT4275/class_1") as f:
    setosa_data = f.read().splitlines()

with open("Iris_TTT4275/class_2") as f:
    versicolor_data = f.read().splitlines()

with open("Iris_TTT4275/class_3") as f:
    virginica_data = f.read().splitlines()

SEPAL = 0
PETAL = 1
N = 50
N_TRAINING = 30

setosa = [] # [Sepal: [Length, width], Petal: [Length, width]] for each element

for flower in setosa_data:
    flower = flower.split(',')
    setosa.append([[float(flower[0]), float(flower[1])], [float(flower[2]), float(flower[3])]])

print(setosa)

# setosa[0][SEPAL] = [length of sepal of first sample, width of sepal of first sample]
# setosa[0][PETAL] = [length of petal of first sample, width of petal of first sample]

versicolor = [] # [Sepal: [Length, width], Petal: [Length, width]] for each element

for flower in versicolor_data:
    flower = flower.split(',')
    versicolor.append([[float(flower[0]), float(flower[1])], [float(flower[2]), float(flower[3])]])

virginica = []

for flower in virginica_data:
    flower = flower.split(',')
    virginica.append([[float(flower[0]), float(flower[1])], [float(flower[2]), float(flower[3])]])

### ------------------------------
### ------------------------------
### -- Here starts the analysis --
### ------------------------------
### ------------------------------

# Plotting the sepal length vs. sepal width for the three classes
'''plt.figure(1)
plt.plot([x[0] for x in setosa[:][SEPAL]], [x[1] for x in setosa[:][SEPAL]], 'ro', label='Setosa')
plt.plot([x[0] for x in versicolor[:][SEPAL]], [x[1] for x in versicolor[:][SEPAL]], 'bo', label='Versicolor')
plt.plot([x[0] for x in virginica[:][SEPAL]], [x[1] for x in virginica[:][SEPAL]], 'go', label='Virginica')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal length vs. sepal width')
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.show()'''

### Task 1a
