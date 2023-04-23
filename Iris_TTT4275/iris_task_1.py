import numpy as np
import matplotlib.pyplot as plt
import copy

# Defining constants

N_CLASSES = 3 # Number of classes
N = 50 # Number of samples in each class

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

def plotting(setosa_set, versicolor_set, virginica_set, title1, title2, name1, name2):
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
    plt.savefig('Plots/Iris_Foerste_Utkast/' + name1 + ".png")

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
    plt.savefig('Plots/Iris_Foerste_Utkast/' + name2 + ".png")
    plt.show()

# plotting(setosa, versicolor, virginica, 'Sepal length vs. sepal width', 'Petal length vs. petal width', 'sepal_l_vs_sepal_w_normalized', 'petal_l_vs_petal_w_normalized')
# plotting(setosa_unormalized, versicolor_unormalized, virginica_unormalized, 'Sepal length vs. sepal width', 'Petal length vs. petal width', 'sepal_l_vs_sepal_w_unormalized', 'petal_l_vs_petal_w_unormalized')

### ------------------------------
### Task 1b

T = [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]] # Target vectors

def training(set_for_training, M = 5000, alpha = 0.3):
    # Creating a weighting matrix and a bias vector, starting with random values between 0 and 1
    w_matrix = np.random.random((N_CLASSES, len(set_for_training[0][0]))) # Weights
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
            # z = np.matmul(w_matrix_bias[0], np.transpose(x_with_bias[0])) + w_matrix_bias[1]*x_with_bias[1]
            # g = 1/(1+np.exp(-z))
            g = sigmoid(np.matmul(w_matrix_bias[0], np.transpose(x_with_bias[0])) + w_matrix_bias[1]*x_with_bias[1])
            u = np.multiply(np.multiply((g-t), g), (1-g))
            e = [np.outer(u, x_with_bias[0]), u*x_with_bias[1]]
            mse_matrix_gradient[0] += e[0] # Adding the error of the weights
            mse_matrix_gradient[1] += e[1] # Adding the error of the bias
        w_matrix_bias[0] = w_matrix_bias[0] - alpha*mse_matrix_gradient[0]
        w_matrix_bias[1] = w_matrix_bias[1] - alpha*mse_matrix_gradient[1]
    return w_matrix_bias

# Training the network but updating the weights and bias after each training input
def training_v2(set_for_training, M = 5000, alpha = 0.3):
    # Creating a weighting matrix and a bias vector, starting with random values between 0 and 1
    w_matrix = np.random.random((N_CLASSES, len(set_for_training[0][0]))) # Weights
    w0 = np.random.random(N_CLASSES) # Bias
    for m in range(M):
        np.random.shuffle(set_for_training) # Randomize the training set for each iteration
        # Training the network for all training inputs, shuffled
        for data in set_for_training:
            t = T[data[1]]
            x = data[0]
            g = sigmoid(np.matmul(w_matrix, np.transpose(x)) + w0)
            u = np.multiply(np.multiply((g-t), g), (1-g))

            w_matrix -= alpha*np.outer(u, x) # Updating the weights with the error of the weights
            w0 -= alpha*u # Updating the bias with the error of the bias

    return [w_matrix, w0]

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Training the network but updating the weights and bias after each training input
def training_w_threshold(set_for_training, M = 5000, alphas = [0.02]):
    mse_arrays = []
    for alpha in alphas:
        # Creating a weighting matrix and a bias vector, starting with random values between 0 and 1
        w_matrix = np.random.random((N_CLASSES, len(set_for_training[0][0]))) # Weights
        w0 = np.random.random(N_CLASSES) # Bias

        # Array that holds the MSE for an entire iteration through the training set
        # Used for plotting
        mse_array = []

        threshold = 7.5
        
        i = 0

        for m in range(M):
        # while(True):
            np.random.shuffle(set_for_training) # Randomize the training set for each iteration
            mse = 0
            # Training the network for all training inputs, shuffled
            for data in set_for_training:
                t = T[data[1]]
                x = data[0]
                g = sigmoid(np.matmul(w_matrix, np.transpose(x)) + w0)
                u = np.multiply(np.multiply((g-t), g), (1-g))

                # Adding up the MSE for every training sample
                mse += np.linalg.norm(0.5*np.transpose(g-t)*(g-t))

                w_matrix -= alpha*np.outer(u, x) # Updating the weights with the error of the weights
                w0 -= alpha*u # Updating the bias with the error of the bias
            
            # Adding MSE for this iteration to an array
            mse_array.append(mse)

        mse_arrays.append([mse_array, alpha])
            
        #     i += 1

        #     if (mse < threshold):
        #         break
        
        # print(i)

    # Plotting the MSE for all learning rates
    for i in range(len(mse_arrays)):
        plt.plot(mse_arrays[i][0], label=str(mse_arrays[i][1]))
    plt.title("MSE for different learning rates")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("Iteration")
    plt.savefig("Plots/Iris_Foerste_Utkast/mse_learning_rates.png")
    plt.show()

    return [w_matrix, w0]

### ------------------------------

# Training the network for all training inputs for M iterations
# iterations = 3000
iterations = 2000
learning_rate = 0.3
# learning_rate = 0.025

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

def plotting_confusion_matrix(confusion_matrix, title, name):
    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap="seismic")
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(["Setosa", "Versicolor", "Virginica"])
    ax.set_yticklabels(["Setosa", "Versicolor", "Virginica"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="w")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    plt.savefig(name)
    plt.show()

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

    # weights = training(training_set, iterations, learning_rate)
    weights = training_v2(training_set, iterations, learning_rate)
    # alphas = [0.0025, 0.005, 0.25, 0.3, 0.6, 1, 1.3]
    # weights = training_w_threshold(training_set, iterations, alphas)

    confusion_matrix_testing, wrong_testing = testing(testing_set, weights)
    confusion_matrix_training, wrong_training = testing(training_set, weights)

    print("Using first 30 samples for training, 20 last samples for testing")

    print(f"Confusion matrix for test-set: \n{confusion_matrix_testing}")
    print(f"Confusion matrix for train-set: \n{confusion_matrix_training}")

    error_rate_testing = wrong_testing/len(testing_set)
    error_rate_training = wrong_training/len(training_set)
    print(f"Error rate for test-set: {error_rate_testing}")
    print(f"Error rate for training-set: {error_rate_training}\n")

    # plotting_confusion_matrix(confusion_matrix_testing, "Confusion matrix for the test-set, first 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_first_testing.png")
    # plotting_confusion_matrix(confusion_matrix_training, "Confusion matrix for the training-set, first 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_first_training.png")

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

    weights = training_v2(training_set, iterations, learning_rate)

    confusion_matrix_testing, wrong_testing = testing(testing_set, weights)
    confusion_matrix_training, wrong_training = testing(training_set, weights)

    print("Using last 30 samples for training, 20 first samples for testing")

    # print(f"Wrong: {wrong}, Total: {len(testing_set)}")
    print(f"Confusion matrix for test-set: \n{confusion_matrix_testing}")
    print(f"Confusion matrix for train-set: \n{confusion_matrix_training}")

    error_rate_testing = wrong_testing/len(testing_set)
    error_rate_training = wrong_training/len(training_set)
    print(f"Error rate for test-set: {error_rate_testing}")
    print(f"Error rate for training-set: {error_rate_training}\n")

    # plotting_confusion_matrix(confusion_matrix_testing, "Confusion matrix for the test-set, last 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_last_testing.png")
    # plotting_confusion_matrix(confusion_matrix_training, "Confusion matrix for the training-set, last 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_last_training.png")

training_30_first_samples()
# training_30_last_samples()
