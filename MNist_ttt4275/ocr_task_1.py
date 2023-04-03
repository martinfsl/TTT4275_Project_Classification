import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

all_data = sio.loadmat('MNist_ttt4275/data_all.mat')

#print(all_data.keys())
#print(all_data.values())

col_size = all_data.get('col_size')[0][0] # Need to add indexes because of the way the data is stored and that we want a scalar
row_size = all_data.get('row_size')[0][0]
num_test = all_data.get('num_test')[0][0]
num_train = all_data.get('num_train')[0][0]

test_labels = all_data.get('testlab')
test_images = all_data.get('testv')

train_labels = all_data.get('trainlab')
train_images = all_data.get('trainv')

#gv = np.reshape(train_images[0], (28, 28))

def plotting(array, rows, cols, title):
    plot_data = np.reshape(array, (rows, cols))
    plt.gray()
    plt.imshow(plot_data)
    plt.title(title)
    plt.show()

#plotting(train_images[6], row_size, col_size, str(train_labels[6][0]))