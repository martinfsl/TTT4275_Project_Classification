import numpy as np
import matplotlib.pyplot as plt
import copy

# Defining constants

N_CLASSES = 3
N_FEATURES = 4
N = 50

# Load the data for the Iris classes
# The data lines are stores in the order: sepal length, sepal width, petal length, petal width - All in cm
setosa = np.genfromtxt("Iris_TTT4275/class_1", delimiter=",")
versicolor = np.genfromtxt("Iris_TTT4275/class_2", delimiter=",")
virginica = np.genfromtxt("Iris_TTT4275/class_3", delimiter=",")

all_samples = np.vstack((setosa, versicolor, virginica))

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
    plt.hist(features[0], bins=20, label='Setosa')
    plt.hist(features[1], bins=20, label='Versicolor')
    plt.hist(features[2], bins=20, label='Virginica')
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
    # Creating the training and testing sets
    # Looks to be a good idea to use the petal length and petal width as features
    # Since we only remove one, sepal length is better than sepal width
    # Thus we use the elements 0, 2 and 3 of the arrays
    training_set_3_features = [[setosa_sample, 0] for setosa_sample in setosa[:N_TRAINING, [0, 2, 3]]]
    training_set_3_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[:N_TRAINING, [0, 2, 3]]]
    training_set_3_features += [[virginica_sample, 2] for virginica_sample in virginica[:N_TRAINING, [0, 2, 3]]]

    testing_set_3_features = [[setosa_sample, 0] for setosa_sample in setosa[N_TRAINING:, [0, 2, 3]]]
    testing_set_3_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[N_TRAINING:, [0, 2, 3]]]
    testing_set_3_features += [[virginica_sample, 2] for virginica_sample in virginica[N_TRAINING:, [0, 2, 3]]]

    weight_3_features = training(training_set_3_features, iterations, learning_rate)
    confusion_matrix_3_features, wrong_3_features = testing(testing_set_3_features, weight_3_features)
    print(f"Confusion matrix for 3 features:\n{confusion_matrix_3_features}")
    print(f"Wrong predictions for 3 features: {wrong_3_features}")
    print(f"Error rate: {wrong_3_features/len(testing_set_3_features)}\n")

### ------------------------------

def two_features():
    # Now removing two features (sepal length and sepal width)
    # Thus we use the elements 2 and 3 of the arrays
    training_set_2_features = [[setosa_sample, 0] for setosa_sample in setosa[:N_TRAINING, [2, 3]]]
    training_set_2_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[:N_TRAINING, [2, 3]]]
    training_set_2_features += [[virginica_sample, 2] for virginica_sample in virginica[:N_TRAINING, [2, 3]]]

    testing_set_2_features = [[setosa_sample, 0] for setosa_sample in setosa[N_TRAINING:, [2, 3]]]
    testing_set_2_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[N_TRAINING:, [2, 3]]]
    testing_set_2_features += [[virginica_sample, 2] for virginica_sample in virginica[N_TRAINING:, [2, 3]]]

    weight_2_features = training(training_set_2_features, iterations, learning_rate)
    confusion_matrix_2_features, wrong_2_features = testing(testing_set_2_features, weight_2_features)
    print(f"Confusion matrix for 2 features:\n{confusion_matrix_2_features}")
    print(f"Wrong predictions for 2 features: {wrong_2_features}")
    print(f"Error rate: {wrong_2_features/len(testing_set_2_features)}\n")

### ------------------------------

def one_feature():
    # Now only using one feature (petal length)
    # Thus we use element 2 of the arrays
    training_set_1_features = [[setosa_sample, 0] for setosa_sample in setosa[:N_TRAINING, [2]]]
    training_set_1_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[:N_TRAINING, [2]]]
    training_set_1_features += [[virginica_sample, 2] for virginica_sample in virginica[:N_TRAINING, [2]]]

    testing_set_1_features = [[setosa_sample, 0] for setosa_sample in setosa[N_TRAINING:, [2]]]
    testing_set_1_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[N_TRAINING:, [2]]]
    testing_set_1_features += [[virginica_sample, 2] for virginica_sample in virginica[N_TRAINING:, [2]]]

    weight_1_features = training(training_set_1_features, iterations, learning_rate)
    confusion_matrix_1_features, wrong_1_features = testing(testing_set_1_features, weight_1_features)
    print(f"Confusion matrix for 1 feature:\n{confusion_matrix_1_features}")
    print(f"Wrong predictions for 1 feature: {wrong_1_features}")
    print(f"Error rate: {wrong_1_features/len(testing_set_1_features)}\n")

### ------------------------------

three_features()
two_features()
one_feature()