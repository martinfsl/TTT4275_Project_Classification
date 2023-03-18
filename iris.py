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

setosa_sepal = [] # [Length, width] for each element
setosa_petal = [] # [Length, width] for each element

for flower in setosa_data:
    flower = flower.split(',')
    setosa_sepal.append([float(flower[0]), float(flower[1])])
    setosa_petal.append([float(flower[2]), float(flower[3])])

versicolor_sepal = [] # [Length, width] for each element
versicolor_petal = [] # [Length, width] for each element

for flower in versicolor_data:
    flower = flower.split(',')
    versicolor_sepal.append([float(flower[0]), float(flower[1])])
    versicolor_petal.append([float(flower[2]), float(flower[3])])

virginica_sepal = []
virginica_petal = []

for flower in virginica_data:
    flower = flower.split(',')
    virginica_sepal.append([float(flower[0]), float(flower[1])])
    virginica_petal.append([float(flower[2]), float(flower[3])])

### ------------------------------
### ------------------------------
### -- Here starts the analysis --
### ------------------------------
### ------------------------------

# Plotting the sepal length vs. sepal width for the three classes
plt.figure(1)
plt.plot([x[0] for x in setosa_sepal], [x[1] for x in setosa_sepal], 'ro', label='Setosa')
plt.plot([x[0] for x in versicolor_sepal], [x[1] for x in versicolor_sepal], 'bo', label='Versicolor')
plt.plot([x[0] for x in virginica_sepal], [x[1] for x in virginica_sepal], 'go', label='Virginica')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal length vs. sepal width')
plt.legend(['Setosa', 'Versicolor', 'Virginica'])
plt.show()
