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

    weights = training(training_set, iterations, learning_rate)

    confusion_matrix, wrong = testing(testing_set, weights)

    print("Using first 30 samples for training, 20 last samples for testing")
    print(f"Wrong: {wrong}, Total: {len(testing_set)}")
    print(f"Confusion matrix: \n{confusion_matrix}")
    error_rate = wrong/len(testing_set)
    print(f"Error rate: {error_rate}\n")

    plotting_confusion_matrix(confusion_matrix, "Confusion matrix when using the first 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_first_samples.png")


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

    plotting_confusion_matrix(confusion_matrix, "Confusion matrix when using the last 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_last_samples.png")

training_30_first_samples()
training_30_last_samples()
