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

# Normalizing the data

all_samples = []

for element in setosa:
    all_samples.append(element)
for element in versicolor:
    all_samples.append(element)
for element in virginica:
    all_samples.append(element)

max_sepal_length = max([x[0] for x in all_samples])
min_sepal_length = min([x[0] for x in all_samples])
max_sepal_width = max([x[1] for x in all_samples])
min_sepal_width = min([x[1] for x in all_samples])

max_petal_length = max([x[2] for x in all_samples])
min_petal_length = min([x[2] for x in all_samples])
max_petal_width = max([x[3] for x in all_samples])
min_petal_width = min([x[3] for x in all_samples])

for flower in setosa:
    flower[0] = (flower[0] - min_sepal_length)/(max_sepal_length - min_sepal_length)
    flower[1] = (flower[1] - min_sepal_width)/(max_sepal_width - min_sepal_width)
    flower[2] = (flower[2] - min_petal_length)/(max_petal_length - min_petal_length)
    flower[3] = (flower[3] - min_petal_width)/(max_petal_width - min_petal_width)

for flower in versicolor:
    flower[0] = (flower[0] - min_sepal_length)/(max_sepal_length - min_sepal_length)
    flower[1] = (flower[1] - min_sepal_width)/(max_sepal_width - min_sepal_width)
    flower[2] = (flower[2] - min_petal_length)/(max_petal_length - min_petal_length)
    flower[3] = (flower[3] - min_petal_width)/(max_petal_width - min_petal_width)

for flower in virginica:
    flower[0] = (flower[0] - min_sepal_length)/(max_sepal_length - min_sepal_length)
    flower[1] = (flower[1] - min_sepal_width)/(max_sepal_width - min_sepal_width)
    flower[2] = (flower[2] - min_petal_length)/(max_petal_length - min_petal_length)
    flower[3] = (flower[3] - min_petal_width)/(max_petal_width - min_petal_width)

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

#w_matrix = 0.5*np.ones((N_CLASSES, np.shape(setosa)[1])) # Weights for the three classes and all features
#w0 = 0.5*np.ones(N_CLASSES) # Bias

w_matrix = np.random.random((N_CLASSES, np.shape(setosa)[1])) # Weights for the three classes and all features
w0 = np.random.random(N_CLASSES) # Bias
# The discriminant vector will be [W w0][x^T 1]^T

w_matrix_bias = [w_matrix, w0]

T = [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]] # Target vectors


#print(np.matmul(w_matrix, setosa_training[0]))

# Training the network for all training inputs for M iterations
M = 5000

training_set = []

for setosa_data in setosa_training:
    training_set.append([setosa_data, 0])
for versicolor_data in versicolor_training:
    training_set.append([versicolor_data, 1])
for virginica_data in virginica_training:
    training_set.append([virginica_data, 2])

alpha = 0.3

#print("w_matrix before: ", w_matrix_bias)

for m in range(M):

    # Randomize the training set for each iteration
    np.random.shuffle(training_set)

    mse_matrix_gradient = [np.zeros((N_CLASSES, np.shape(setosa)[1])), np.zeros(N_CLASSES)] # MSE for the three classes and all features
    
    # Training the network for all training inputs, shuffled
    for data in training_set:

        t = T[data[1]]
        x = data[0]

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

    w_matrix_bias[0] = w_matrix_bias[0] - alpha*mse_matrix_gradient[0]
    w_matrix_bias[1] = w_matrix_bias[1] - alpha*mse_matrix_gradient[1]

    print(f"Iteration: {m+1}")

    #print(f"w_matrix after {m+1} iterations: {w_matrix_bias}")

# Creating a set for testing
testing_set = []

for setosa_data in setosa_testing:
    testing_set.append([setosa_data, 0])
for versicolor_data in versicolor_testing:
    testing_set.append([versicolor_data, 1])
for virginica_data in virginica_testing:
    testing_set.append([virginica_data, 2])

np.random.shuffle(testing_set)

wrong = 0

# confusion_matrix[true_class][predicted_class]
confusion_matrix = np.zeros((N_CLASSES, N_CLASSES))

for test_sample in testing_set:
    true_class = test_sample[1]

    sample = [np.transpose(test_sample[0]), 1]
    g = 1/(1+np.exp(-np.matmul(w_matrix_bias[0], sample[0]) - w_matrix_bias[1]*sample[1]))
    predicted_class = np.argmax(g)

    confusion_matrix[true_class][predicted_class] += 1

    if predicted_class != true_class:
        wrong += 1
    
print(f"Wrong: {wrong}, Total: {len(testing_set)}")
print(f"Confusion matrix: \n{confusion_matrix}")