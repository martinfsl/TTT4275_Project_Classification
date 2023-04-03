import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# Dictionary with all the data
all_data = sio.loadmat('MNist_ttt4275/data_all.mat')

# Defining constants from the data dictionary (all_data)
# Need to add indexes because of the way the data is stored and that we want a scalar
COL_SIZE, ROW_SIZE = all_data.get('col_size')[0][0], all_data.get('row_size')[0][0]
NUM_TEST, NUM_TRAIN = all_data.get('num_test')[0][0], all_data.get('num_train')[0][0]
N = all_data.get('vec_size')[0][0]
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

#plotting(train_images[6], row_size, col_size, str(train_labels[6][0]))

T = np.identity(NUMBERS) # Target vectors

# Create training and testing data
training = []
for i in range(NUM_TRAIN):
    training.append([train_images[i], train_labels[i][0]])
testing = []
for i in range(NUM_TEST):
    testing.append([test_images[i], test_labels[i][0]])


