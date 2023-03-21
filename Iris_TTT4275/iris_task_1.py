import numpy as np
import matplotlib.pyplot as plt

# Defining constants

N_CLASSES = 3
N = 50
N_TRAINING = 30

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

'''
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
'''

### Task 1a

setosa_training = setosa[:N_TRAINING]
setosa_testing = setosa[N_TRAINING:]

versicolor_training = versicolor[:N_TRAINING]
versicolor_testing = versicolor[N_TRAINING:]

virginica_training = virginica[:N_TRAINING]
virginica_testing = virginica[N_TRAINING:]

### Task 1b

# Creating a weighting matrix and a bias vector

w_matrix = np.ones((N_CLASSES, np.shape(setosa)[1])) # Weights for the three classes and all features
w0 = np.ones(N_CLASSES) # Bias
# The discriminant vector will be [W w0][x^T 1]^T

w_matrix_bias = [w_matrix, w0]

T = [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]] # Target vectors


#print(np.matmul(w_matrix, setosa_training[0]))

# Training the network for all training inputs for 100 iterations
M = 100

#training_set = []

#for setosa in setosa_training:
#    training_set.append([setosa, 0])
#for versicolor in versicolor_training:
#    training_set.append([versicolor, 1])
#for virginica in virginica_training:
#    training_set.append([virginica, 2])

# Randomize the training set
#np.random.shuffle(training_set)

alpha = 0.02

for m in range(M):

    mse_matrix_gradient = [np.zeros((N_CLASSES, np.shape(setosa)[1])), np.zeros(N_CLASSES)] # MSE for the three classes and all features
        
    # Training the network for all training inputs, this is one iteration
    for number in range(N_TRAINING):
        if m < M/3:
            t = T[0]
            x = setosa_training[number]
        elif M/3 <= m < 2*M/3:
            t = T[1]
            x = versicolor_training[number]
        elif 2*M/3 <= m:
            t = T[2]
            x = virginica_training[number]

        x_with_bias = [np.transpose(x), 1]

        #print("x: ", x_with_bias)

        z = np.matmul(w_matrix_bias[0], np.transpose(x_with_bias[0])) + w_matrix_bias[1]*x_with_bias[1]
        g = 1/(1+np.exp(-z))
        #u = np.multiply((g-t), g, (1-g))
        u = np.multiply(np.multiply((g-t), g), (1-g))

        #print("z: ", z)
        #print("g: ", g)
        #print("u: ", u)

        # Need change below here:

        e = [np.outer(u, x_with_bias[0]), u*x_with_bias[1]]
        #print("e, ", e)
        #print("mse before, ", mse_matrix_gradient)
        mse_matrix_gradient[0] += e[0] # Adding the error of the weights
        mse_matrix_gradient[1] += e[1] # Adding the error of the bias

        #print("mse_gradient: ", mse_matrix_gradient)

    print("w_matrix before: ", w_matrix_bias)

    w_matrix_bias[0] = w_matrix_bias[0] - alpha*mse_matrix_gradient[0]
    w_matrix_bias[1] = w_matrix_bias[1] - alpha*mse_matrix_gradient[1]

    print("w_matrix after: ", w_matrix_bias)

    test = setosa_testing[0]
    test_with_bias = [np.transpose(test), 1]
    z = np.matmul(w_matrix_bias[0], np.transpose(test_with_bias[0])) + w_matrix_bias[1]*test_with_bias[1]
    g = 1/(1+np.exp(-z))
    print("g: ", g, " class: ", np.argmax(g))

# Arange so that the training set comes in random order, not a structured one
