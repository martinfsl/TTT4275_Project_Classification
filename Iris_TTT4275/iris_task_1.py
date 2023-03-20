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

setosa = [] # [Sepal length, Sepal width, Petal length, Petal width] for each element

for flower in setosa_data:
    flower = flower.split(',')
    setosa.append([float(flower[0]), float(flower[1]), float(flower[2]), float(flower[3])])

versicolor = [] # [Sepal length, Sepal width, Petal length, Petal width] for each element

for flower in versicolor_data:
    flower = flower.split(',')
    versicolor.append([float(flower[0]), float(flower[1]), float(flower[2]), float(flower[3])])

virginica = [] # [Sepal length, Sepal width, Petal length, Petal width] for each element

for flower in virginica_data:
    flower = flower.split(',')
    virginica.append([float(flower[0]), float(flower[1]), float(flower[2]), float(flower[3])])

### ------------------------------
### ------------------------------
### -- Here starts the analysis --
### ------------------------------
### ------------------------------

# Plotting the sepal length vs. sepal width for the three classes
plt.figure(1)
plt.plot([x[0] for x in setosa], [x[1] for x in setosa], 'ro', label='Setosa')
plt.plot([x[0] for x in versicolor], [x[1] for x in versicolor], 'bo', label='Versicolor')
plt.plot([x[0] for x in virginica], [x[1] for x in virginica], 'go', label='Virginica')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal length vs. sepal width')
plt.legend(['Setosa', 'Versicolor', 'Virginica'])

# Plotting the petal length vs. petal width for the three classes
plt.figure(2)
plt.plot([x[2] for x in setosa], [x[3] for x in setosa], 'ro', label='Setosa')
plt.plot([x[2] for x in versicolor], [x[3] for x in versicolor], 'bo', label='Versicolor')
plt.plot([x[2] for x in virginica], [x[3] for x in virginica], 'go', label='Virginica')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title('Petal length vs. petal width')
plt.legend(['Setosa', 'Versicolor', 'Virginica'])

plt.show()

### Task 1a
N = 50
N_TRAINING = 30

setosa_training = setosa[:N_TRAINING]
setosa_testing = setosa[N_TRAINING:]

versicolor_training = versicolor[:N_TRAINING]
versicolor_testing = versicolor[N_TRAINING:]

virginica_training = virginica[:N_TRAINING]
virginica_testing = virginica[N_TRAINING:]

