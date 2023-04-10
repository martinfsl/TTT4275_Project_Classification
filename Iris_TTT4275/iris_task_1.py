import numpy as np
import matplotlib.pyplot as plt
import copy

# Defining constants

N_CLASSES = 3
N = 50

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

# Collecting all the samples in one list (Used for normalization)
all_samples = []
for i in range(N):
    all_samples.append(setosa[i])
    all_samples.append(versicolor[i])
    all_samples.append(virginica[i])

'''
Ex. of finding the max and min of a specific feature
max_sepal_length = max([x[0] for x in all_samples])
min_sepal_length = min([x[0] for x in all_samples])
'''

# Want to keep the original data for plotting later
setosa_unormalized, versicolor_unormalized, virginica_unormalized = copy.deepcopy(setosa), copy.deepcopy(versicolor), copy.deepcopy(virginica)

def normalization(flower_set, samples):
    for flower in flower_set:  
        flower[0] = (flower[0] - min([x[0] for x in all_samples]))/(max([x[0] for x in all_samples]) - min([x[0] for x in all_samples]))
        flower[1] = (flower[1] - min([x[1] for x in all_samples]))/(max([x[1] for x in all_samples]) - min([x[1] for x in all_samples]))
        flower[2] = (flower[2] - min([x[2] for x in all_samples]))/(max([x[2] for x in all_samples]) - min([x[2] for x in all_samples]))
        flower[3] = (flower[3] - min([x[3] for x in all_samples]))/(max([x[3] for x in all_samples]) - min([x[3] for x in all_samples]))

normalization(setosa, all_samples)
normalization(versicolor, all_samples)
normalization(virginica, all_samples)

### ------------------------------
### ------------------------------
### -- Here starts the analysis --
### ------------------------------
### ------------------------------

def plotting(setosa_set, versicolor_set, virginica_set, title1, title2):
    # Plotting the sepal length vs. sepal width for the three classes
    plt.figure(1)
    plt.plot([x[0] for x in setosa_set], [x[1] for x in setosa_set], 'ro', label='Setosa')
    plt.plot([x[0] for x in versicolor_set], [x[1] for x in versicolor_set], 'bo', label='Versicolor')
    plt.plot([x[0] for x in virginica_set], [x[1] for x in virginica_set], 'go', label='Virginica')
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)')
    #plt.title('Sepal length vs. sepal width')
    plt.title(title1)
    plt.legend(['Setosa', 'Versicolor', 'Virginica'])

    # Plotting the petal length vs. petal width for the three classes
    plt.figure(2)
    plt.plot([x[2] for x in setosa_set], [x[3] for x in setosa_set], 'ro', label='Setosa')
    plt.plot([x[2] for x in versicolor_set], [x[3] for x in versicolor_set], 'bo', label='Versicolor')
    plt.plot([x[2] for x in virginica_set], [x[3] for x in virginica_set], 'go', label='Virginica')
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Petal width (cm)')
    #plt.title('Petal length vs. petal width')
    plt.title(title2)
    plt.legend(['Setosa', 'Versicolor', 'Virginica'])

    plt.show()

#plotting(setosa, versicolor, virginica, 'Sepal length vs. sepal width', 'Petal length vs. petal width')
#plotting(setosa_unormalized, versicolor_unormalized, virginica_unormalized, 'Sepal length vs. sepal width', 'Petal length vs. petal width')

### ------------------------------
### Task 1b

T = [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]] # Target vectors

def training(set_for_training, M = 5000, alpha = 0.3):
    # Creating a weighting matrix and a bias vector
    w_matrix = np.random.random((N_CLASSES, len(set_for_training[0][0]))) # Weights for the three classes and all features
    w0 = np.random.random(N_CLASSES) # Bias
    # The discriminant vector will be [W w0][x^T 1]^T
    w_matrix_bias = [w_matrix, w0]

    for m in range(M):
        np.random.shuffle(set_for_training) # Randomize the training set for each iteration
        mse_matrix_gradient = [np.zeros((N_CLASSES, len(set_for_training[0][0]))), np.zeros(N_CLASSES)] # MSE for the three classes and all features
        
        # Training the network for all training inputs, shuffled
        for data in set_for_training:
            t = T[data[1]]
            x = data[0]
            x_with_bias = [np.transpose(x), 1]

            z = np.matmul(w_matrix_bias[0], np.transpose(x_with_bias[0])) + w_matrix_bias[1]*x_with_bias[1]
            g = 1/(1+np.exp(-z))
            u = np.multiply(np.multiply((g-t), g), (1-g))
            e = [np.outer(u, x_with_bias[0]), u*x_with_bias[1]]

            mse_matrix_gradient[0] += e[0] # Adding the error of the weights
            mse_matrix_gradient[1] += e[1] # Adding the error of the bias

        w_matrix_bias[0] = w_matrix_bias[0] - alpha*mse_matrix_gradient[0]
        w_matrix_bias[1] = w_matrix_bias[1] - alpha*mse_matrix_gradient[1]

        #print(f"Iteration: {m+1}")
    return w_matrix_bias

### ------------------------------

# Training the network for all training inputs for M iterations
iterations = 3000
learning_rate = 0.3

def testing(testing_set, weights):
    confusion_matrix = np.zeros((N_CLASSES, N_CLASSES))
    wrong = 0

    for test_sample in testing_set:
        true_class = test_sample[1]
        sample = [np.transpose(test_sample[0]), 1]
        g = 1/(1+np.exp(-np.matmul(weights[0], sample[0]) - weights[1]*sample[1]))
        predicted_class = np.argmax(g)
        confusion_matrix[true_class][predicted_class] += 1
        if predicted_class != true_class:
            wrong += 1
    return confusion_matrix, wrong

# Using the first 30 samples for training and the last 20 for testing
def training_30_first_samples():
    N_TRAINING = 30
    # Creating a set for training
    training_set = []
    for setosa_data in setosa[:N_TRAINING]: training_set.append([setosa_data, 0])
    for versicolor_data in versicolor[:N_TRAINING]: training_set.append([versicolor_data, 1])
    for virginica_data in virginica[:N_TRAINING]: training_set.append([virginica_data, 2])
    # Creating a set for testing
    testing_set = []
    for setosa_data in setosa[N_TRAINING:]: testing_set.append([setosa_data, 0])
    for versicolor_data in versicolor[N_TRAINING:]: testing_set.append([versicolor_data, 1])
    for virginica_data in virginica[N_TRAINING:]: testing_set.append([virginica_data, 2])

    weights = training(training_set, iterations, learning_rate)

    confusion_matrix, wrong = testing(testing_set, weights)

    print("Using first 30 samples for training, 20 last samples for testing")
    print(f"Wrong: {wrong}, Total: {len(testing_set)}")
    print(f"Confusion matrix: \n{confusion_matrix}")
    error_rate = wrong/len(testing_set)
    print(f"Error rate: {error_rate}\n")


# Using the last 30 samples for training and the first 20 for testing
def training_30_last_samples():
    N_TESTING = 20
    # Creating a set for training
    training_set = []
    for setosa_data in setosa[N_TESTING:]: training_set.append([setosa_data, 0])
    for versicolor_data in versicolor[N_TESTING:]: training_set.append([versicolor_data, 1])
    for virginica_data in virginica[N_TESTING:]: training_set.append([virginica_data, 2])
    # Creating a set for testing
    testing_set = []
    for setosa_data in setosa[:N_TESTING]: testing_set.append([setosa_data, 0])
    for versicolor_data in versicolor[:N_TESTING]: testing_set.append([versicolor_data, 1])
    for virginica_data in virginica[:N_TESTING]: testing_set.append([virginica_data, 2])

    weights = training(training_set, iterations, learning_rate)

    confusion_matrix, wrong = testing(testing_set, weights)

    print("Using last 30 samples for training, 20 first samples for testing")
    print(f"Wrong: {wrong}, Total: {len(testing_set)}")
    print(f"Confusion matrix: \n{confusion_matrix}")
    error_rate = wrong/len(testing_set)
    print(f"Error rate: {error_rate}\n")

training_30_first_samples()
training_30_last_samples()

