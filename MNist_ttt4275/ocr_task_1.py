import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

all_data = sio.loadmat('MNist_ttt4275/data_all.mat')

#print(all_data.keys())
#print(all_data.values())

COL_SIZE = all_data.get('col_size')[0][0] # Need to add indexes because of the way the data is stored and that we want a scalar
ROW_SIZE = all_data.get('row_size')[0][0]
NUM_TEST = all_data.get('num_test')[0][0]
NUM_TRAIN = all_data.get('num_train')[0][0]
N = all_data.get('vec_size')[0][0]
NUMBERS = 10 # 0-9

test_labels = all_data.get('testlab')
test_images = all_data.get('testv')

train_labels = all_data.get('trainlab')
train_images = all_data.get('trainv')

#gv = np.reshape(train_images[0], (28, 28))

def plotting(array, rows = ROW_SIZE, cols = COL_SIZE, title = ""):
    plot_data = np.reshape(array, (rows, cols))
    plt.gray()
    plt.imshow(plot_data)
    plt.title(title)
    plt.show()

#plotting(train_images[6], row_size, col_size, str(train_labels[6][0]))

T = np.identity(NUMBERS) # Target vectors

training = []
for i in range(NUM_TRAIN):
    training.append([train_images[i], train_labels[i][0]])

testing = []
for i in range(NUM_TEST):
    testing.append([test_images[i], test_labels[i][0]])


