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

def plotting(setosa_set, versicolor_set, virginica_set, title1, title2, name1, name2):
    # Plotting the sepal width vs. petal width for the three classes
    plt.figure(1)
    plt.plot([x[1] for x in setosa_set], [x[3] for x in setosa_set], 'ro', label='Setosa')
    plt.plot([x[1] for x in versicolor_set], [x[3] for x in versicolor_set], 'bo', label='Versicolor')
    plt.plot([x[1] for x in virginica_set], [x[3] for x in virginica_set], 'go', label='Virginica')
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Petal width (cm)')
    plt.title('Petal length vs. petal width')
    plt.title(title1)
    plt.legend(['Setosa', 'Versicolor', 'Virginica'])
    # name = 'petal_l_vs_petal_w_normalized'
    # plt.savefig('Plots/Iris_Features/' + name + ".png")
    plt.show()

# plotting(setosa, versicolor, virginica, 'Sepal length vs. sepal width', 'Petal length vs. petal width', 'sepal_l_vs_sepal_w_normalized', 'petal_l_vs_petal_w_normalized')
# plotting(setosa_unormalized, versicolor_unormalized, virginica_unormalized, 'Sepal length vs. sepal width', 'Petal length vs. petal width', 'sepal_l_vs_sepal_w_unormalized', 'petal_l_vs_petal_w_unormalized')

T = [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]] # Target vectors

# Training the network but updating the weights and bias after each training input
def training(set_for_training, M = 5000, alpha = 0.3):
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

# Training the network for all training inputs for M iterations
iterations = 1000
learning_rate = 0.6

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
    # plt.savefig(name)
    plt.show()

#---------------------------
# Task 1
#---------------------------

print("Starting task 1")

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

    confusion_matrix_testing, wrong_testing = testing(testing_set, weights)
    confusion_matrix_training, wrong_training = testing(training_set, weights)

    print("Using first 30 samples for training, 20 last samples for testing")

    print(f"Confusion matrix for test-set: \n{confusion_matrix_testing}")
    print(f"Confusion matrix for train-set: \n{confusion_matrix_training}")

    error_rate_testing = wrong_testing/len(testing_set)
    error_rate_training = wrong_training/len(training_set)
    print(f"Error rate for test-set: {error_rate_testing}")
    print(f"Error rate for training-set: {error_rate_training}\n")

    plotting_confusion_matrix(confusion_matrix_testing, "Confusion matrix for the test-set, first 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_first_testing.png")
    plotting_confusion_matrix(confusion_matrix_training, "Confusion matrix for the training-set, first 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_first_training.png")

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

    plotting_confusion_matrix(confusion_matrix_testing, "Confusion matrix for the test-set, last 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_last_testing.png")
    plotting_confusion_matrix(confusion_matrix_training, "Confusion matrix for the training-set, last 30 samples for training", "Plots/Iris_Foerste_Utkast/Confusion_matrix_30_last_training.png")

training_30_first_samples()
training_30_last_samples()

print("Finished task 1\n")

#---------------------------
# Task 2
#---------------------------

print("Starting task 2")

def separating_and_plotting():
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

    plot_histograms([setosa_features[0], versicolor_features[0], virginica_features[0]], 'Sepal length', 'Plots/Iris_Foerste_Utkast/sepallength.png')
    plot_histograms([setosa_features[1], versicolor_features[1], virginica_features[1]], 'Sepal width', 'Plots/Iris_Foerste_Utkast/sepalwidth.png')
    plot_histograms([setosa_features[2], versicolor_features[2], virginica_features[2]], 'Petal length', 'Plots/Iris_Foerste_Utkast/petallength.png')
    plot_histograms([setosa_features[3], versicolor_features[3], virginica_features[3]], 'Petal width', 'Plots/Iris_Foerste_Utkast/petalwidth.png')


# Plot the histogram for the features of the three classes

def plot_histograms(features, title, name):
    plt.figure()
    plt.hist(features[0], bins=19, label='Setosa')
    plt.hist(features[1], bins=19, label='Versicolor')
    plt.hist(features[2], bins=19, label='Virginica')
    plt.xlabel('Size (cm)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    # plt.savefig(name)
    plt.show()

N_TRAINING = 30

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

    # weight_3_features = training(training_set_3_features, iterations, learning_rate)
    weight_3_features = training(training_set_3_features, iterations, learning_rate)

    confusion_matrix_3_features, wrong_3_features = testing(testing_set_3_features, weight_3_features)
    print(f"Confusion matrix for 3 features:\n{confusion_matrix_3_features}")
    print(f"Wrong predictions for 3 features: {wrong_3_features}")
    print(f"Error rate: {wrong_3_features/len(testing_set_3_features)}\n")

    plotting_confusion_matrix(confusion_matrix_3_features, "Confusion matrix for 3 features", "Plots/Iris_Foerste_Utkast/confusion_matrix_3_features.png")

def two_features():
    # Now removing two features (sepal length and sepal width)
    # Thus we use the elements 2 and 3 of the arrays
    training_set_2_features = [[setosa_sample, 0] for setosa_sample in setosa[:N_TRAINING, [2, 3]]]
    training_set_2_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[:N_TRAINING, [2, 3]]]
    training_set_2_features += [[virginica_sample, 2] for virginica_sample in virginica[:N_TRAINING, [2, 3]]]

    testing_set_2_features = [[setosa_sample, 0] for setosa_sample in setosa[N_TRAINING:, [2, 3]]]
    testing_set_2_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[N_TRAINING:, [2, 3]]]
    testing_set_2_features += [[virginica_sample, 2] for virginica_sample in virginica[N_TRAINING:, [2, 3]]]

    # weight_2_features = training(training_set_2_features, iterations, learning_rate)
    weight_2_features = training(training_set_2_features, iterations, learning_rate)

    confusion_matrix_2_features, wrong_2_features = testing(testing_set_2_features, weight_2_features)
    print(f"Confusion matrix for 2 features:\n{confusion_matrix_2_features}")
    print(f"Wrong predictions for 2 features: {wrong_2_features}")
    print(f"Error rate: {wrong_2_features/len(testing_set_2_features)}\n")

    plotting_confusion_matrix(confusion_matrix_2_features, "Confusion matrix for 2 features", "Plots/Iris_Foerste_Utkast/confusion_matrix_2_features.png")

def one_feature():
    # Now only using one feature (petal length)
    # Thus we use element 2 of the arrays
    training_set_1_features = [[setosa_sample, 0] for setosa_sample in setosa[:N_TRAINING, [2]]]
    training_set_1_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[:N_TRAINING, [2]]]
    training_set_1_features += [[virginica_sample, 2] for virginica_sample in virginica[:N_TRAINING, [2]]]

    testing_set_1_features = [[setosa_sample, 0] for setosa_sample in setosa[N_TRAINING:, [2]]]
    testing_set_1_features += [[versicolor_sample, 1] for versicolor_sample in versicolor[N_TRAINING:, [2]]]
    testing_set_1_features += [[virginica_sample, 2] for virginica_sample in virginica[N_TRAINING:, [2]]]

    # weight_1_features = training(training_set_1_features, iterations, learning_rate)
    weight_1_features = training(training_set_1_features, iterations, learning_rate)

    confusion_matrix_1_features, wrong_1_features = testing(testing_set_1_features, weight_1_features)
    print(f"Confusion matrix for 1 feature:\n{confusion_matrix_1_features}")
    print(f"Wrong predictions for 1 feature: {wrong_1_features}")
    print(f"Error rate: {wrong_1_features/len(testing_set_1_features)}\n")

    plotting_confusion_matrix(confusion_matrix_1_features, "Confusion matrix for 1 feature", "Plots/Iris_Foerste_Utkast/confusion_matrix_1_features.png")

three_features()
two_features()
one_feature()

print("Finished task 2\n")
