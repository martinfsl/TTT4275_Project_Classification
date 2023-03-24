import numpy as np
import matplotlib.pyplot as plt
import copy

# Defining constants

N_CLASSES = 3
N_FEATURES = 4
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

# plotting(setosa, versicolor, virginica, 'Sepal length vs. sepal width', 'Petal length vs. petal width')
# plotting(setosa_unormalized, versicolor_unormalized, virginica_unormalized, 'Sepal length vs. sepal width', 'Petal length vs. petal width')

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

### ------------------------------
### Task 2:
### ------------------------------
# First 30 samples for training, last 20 samples for testing

# Getting all features into one array
setosa_features = np.zeros((4, N)) # [[all sepal length], [all sepal width], [all petal length], [all petal width]]
versicolor_features = np.zeros((4, N))
virginica_features = np.zeros((4, N))

for i in range(N):
    setosa_features[0][i], setosa_features[1][i] = setosa_unormalized[i][0], setosa_unormalized[i][1]
    setosa_features[2][i], setosa_features[3][i] = setosa_unormalized[i][2], setosa_unormalized[i][3]

    versicolor_features[0][i], versicolor_features[1][i] = versicolor_unormalized[i][0], versicolor_unormalized[i][1]
    versicolor_features[2][i], versicolor_features[3][i] = versicolor_unormalized[i][2], versicolor_unormalized[i][3]

    virginica_features[0][i], virginica_features[1][i] = virginica_unormalized[i][0], virginica_unormalized[i][1]
    virginica_features[2][i], virginica_features[3][i] = virginica_unormalized[i][2], virginica_unormalized[i][3]

# Plot the histogram for the features of the three classes

def plot_histograms(features, title):
    plt.figure()
    plt.hist(features[0], bins=18, label='Setosa')
    plt.hist(features[1], bins=18, label='Versicolor')
    plt.hist(features[2], bins=18, label='Virginica')
    plt.xlabel('Size (cm)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()

# plot_histograms([setosa_features[0], versicolor_features[0], virginica_features[0]], 'Sepal length')
# plot_histograms([setosa_features[1], versicolor_features[1], virginica_features[1]], 'Sepal width')
# plot_histograms([setosa_features[2], versicolor_features[2], virginica_features[2]], 'Petal length')
# plot_histograms([setosa_features[3], versicolor_features[3], virginica_features[3]], 'Petal width')

N_TRAINING = 30

### ------------------------------

def three_features():
    # Looks to be a good idea to use the petal length and petal width as features
    # Since we only remove one, sepal length is better than sepal width

    setosa_3_features, versicolor_3_features, virginica_3_features = [], [], []

    for i in range(N):
        setosa_3_features.append([setosa[i][0], setosa[i][2], setosa[i][3]])
        versicolor_3_features.append([versicolor[i][0], versicolor[i][2], versicolor[i][3]])
        virginica_3_features.append([virginica[i][0], virginica[i][2], virginica[i][3]])

    training_set_3_features, testing_set_3_features = [], []

    # Creating the training set
    for setosa_data in setosa_3_features[:N_TRAINING]: training_set_3_features.append([setosa_data, 0])
    for versicolor_data in versicolor_3_features[:N_TRAINING]: training_set_3_features.append([versicolor_data, 1])
    for virginica_data in virginica_3_features[:N_TRAINING]: training_set_3_features.append([virginica_data, 2])
    # Creating the testing set
    for setosa_data in setosa_3_features[N_TRAINING:]: testing_set_3_features.append([setosa_data, 0])
    for versicolor_data in versicolor_3_features[N_TRAINING:]: testing_set_3_features.append([versicolor_data, 1])
    for virginica_data in virginica_3_features[N_TRAINING:]: testing_set_3_features.append([virginica_data, 2])

    weight_3_features = training(training_set_3_features, iterations, learning_rate)
    confusion_matrix_3_features, wrong_3_features = testing(testing_set_3_features, weight_3_features)
    print(f"Confusion matrix for 3 features:\n{confusion_matrix_3_features}")
    print(f"Wrong predictions for 3 features: {wrong_3_features}")

### ------------------------------

def two_features():
    # Now removing two features (sepal length and sepal width)
    setosa_2_features, versicolor_2_features, virginica_2_features = [], [], []

    for i in range(N):
        setosa_2_features.append([setosa[i][2], setosa[i][3]])
        versicolor_2_features.append([versicolor[i][2], versicolor[i][3]])
        virginica_2_features.append([virginica[i][2], virginica[i][3]])

    training_set_2_features, testing_set_2_features = [], []

    # Creating the training set
    for setosa_data in setosa_2_features[:N_TRAINING]: training_set_2_features.append([setosa_data, 0])
    for versicolor_data in versicolor_2_features[:N_TRAINING]: training_set_2_features.append([versicolor_data, 1])
    for virginica_data in virginica_2_features[:N_TRAINING]: training_set_2_features.append([virginica_data, 2])
    # Creating the testing set
    for setosa_data in setosa_2_features[N_TRAINING:]: testing_set_2_features.append([setosa_data, 0])
    for versicolor_data in versicolor_2_features[N_TRAINING:]: testing_set_2_features.append([versicolor_data, 1])
    for virginica_data in virginica_2_features[N_TRAINING:]: testing_set_2_features.append([virginica_data, 2])

    weight_2_features = training(training_set_2_features, iterations, learning_rate)
    confusion_matrix_2_features, wrong_2_features = testing(testing_set_2_features, weight_2_features)
    print(f"Confusion matrix for 2 features:\n{confusion_matrix_2_features}")
    print(f"Wrong predictions for 2 features: {wrong_2_features}")

### ------------------------------

def one_feature():
    # Now only using one feature (petal length)
    setosa_1_features, versicolor_1_features, virginica_1_features = [], [], []

    for i in range(N):
        setosa_1_features.append([setosa[i][2]])
        versicolor_1_features.append([versicolor[i][2]])
        virginica_1_features.append([virginica[i][2]])

    training_set_1_features, testing_set_1_features = [], []

    # Creating the training set
    for setosa_data in setosa_1_features[:N_TRAINING]: training_set_1_features.append([setosa_data, 0])
    for versicolor_data in versicolor_1_features[:N_TRAINING]: training_set_1_features.append([versicolor_data, 1])
    for virginica_data in virginica_1_features[:N_TRAINING]: training_set_1_features.append([virginica_data, 2])
    # Creating the testing set
    for setosa_data in setosa_1_features[N_TRAINING:]: testing_set_1_features.append([setosa_data, 0])
    for versicolor_data in versicolor_1_features[N_TRAINING:]: testing_set_1_features.append([versicolor_data, 1])
    for virginica_data in virginica_1_features[N_TRAINING:]: testing_set_1_features.append([virginica_data, 2])

    weight_1_features = training(training_set_1_features, iterations, learning_rate)
    confusion_matrix_1_features, wrong_1_features = testing(testing_set_1_features, weight_1_features)
    print(f"Confusion matrix for 1 feature:\n{confusion_matrix_1_features}")
    print(f"Wrong predictions for 1 feature: {wrong_1_features}")

### ------------------------------