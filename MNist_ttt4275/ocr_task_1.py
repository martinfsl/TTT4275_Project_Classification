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

plotting(train_images[9], ROW_SIZE, COL_SIZE, str(train_labels[9][0]))

# Create training and testing data
training = []
for i in range(NUM_TRAIN):
    training.append([train_images[i], train_labels[i][0]])
testing = []
for i in range(NUM_TEST):
    testing.append([test_images[i], test_labels[i][0]])

wrong = []
i = 0

# Function for calculating the Euclidean distance between two vectors
def calc_distance(array_a, array_b):
    distance = np.matmul(np.transpose(np.array(array_a) - np.array(array_b)), (np.array(array_a) - np.array(array_b)))
    return distance

def classify(sample, templates):
    distances = []
    for template in templates:
        distances.append(calc_distance(sample[0], template[0]))
        #print(distances[-1], template[1])

    min_index = np.argmin(distances)
    # print("Min index: ", min_index)
    predicted_label = templates[min_index][1]
    return predicted_label

def test(samples, templates):
    wrong = 0
    i = 0
    for sample in samples:
        predicted_label = classify(sample, templates)
        if predicted_label != sample[1]:
            wrong += 1
        # print(f"Sample {i} - Predicted: {predicted_label}, Actual: {sample[1]}, Wrong: {wrong}")
        i += 1
    # return wrong

#test(testing[0:5], training)


