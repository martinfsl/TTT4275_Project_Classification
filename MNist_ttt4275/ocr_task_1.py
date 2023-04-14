import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.spatial.distance import euclidean

# Dictionary with all the data
all_data = sio.loadmat('MNist_ttt4275/data_all.mat')

# Defining constants from the data dictionary (all_data)
# Need to add indexes because of the way the data is stored and that we want a scalar
COL_SIZE, ROW_SIZE = all_data.get('col_size')[0][0], all_data.get('row_size')[0][0]
NUM_TEST, NUM_TRAIN = all_data.get('num_test')[0][0], all_data.get('num_train')[0][0]
N = all_data.get('vec_size')[0][0] # Number of pixels in each image - 784

NUMBERS = 10 # 0-9

# Get the data from the dictionary (all_data)
test_labels, test_images = all_data.get('testlab'), all_data.get('testv')
train_labels, train_images = all_data.get('trainlab'), all_data.get('trainv')

# Function for plotting the images of the digits
def plotting(array, rows = ROW_SIZE, cols = COL_SIZE, title = ""):
    plot_data = np.reshape(array, (rows, cols))
    plt.gray()
    plt.imshow(plot_data)
    plt.title(title)
    plt.show()

# plotting(train_images[9], ROW_SIZE, COL_SIZE, str(train_labels[9][0]))

# Create training and testing data
# training = []
# for i in range(NUM_TRAIN):
#     training.append([train_images[i], train_labels[i][0]])
# testing = []
# for i in range(NUM_TEST):
#     testing.append([test_images[i], test_labels[i][0]])

wrong = []
i = 0

# Function for calculating the Euclidean distance between two matrices
def calc_distance(samples, templates):
    distance = np.matmul(templates, np.transpose(samples))
    return distance

def classify(samples, templates):
    
    distances = calc_distance(samples, templates)
    wrong = 0

    for i in range(np.shape(samples)[0]):
        predicted_label = train_labels[np.argmin([np.abs(r[i]) for r in distances])][0]
        true_label = test_labels[i][0]
        # print(predicted_label, true_label)
        if predicted_label != true_label:
            wrong += 1
    return wrong

def myFunc(samples, templates):
    s = classify(samples, templates)
    print(s)

#test(testing[0:5], training)

# Calculate the euclidean distance
# w = np.matmul(train_images, np.transpose(test_images))

myFunc(test_images[:20], train_images)
