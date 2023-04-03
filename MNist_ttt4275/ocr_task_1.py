import numpy as np
import matplotlib.pyplot as plt

NX = NY = 28
N = NX*NY

labels = []
samples = []

# Load the test data from the file MNist_ttt4275/test_images.bin which is a binary file
# with 10000 images of size 28x28
with open('MNist_ttt4275/train_images.bin', 'rb') as f:
    for i in range(60000):
        samples.append(np.fromfile(f, dtype=np.uint8, count=N))

# Read the labels from the file MNist_ttt4275/test_labels.bin which is a binary file
with open('MNist_ttt4275/train_labels.bin', 'rb') as f:
    for i in range(60000):
        label = np.fromfile(f, dtype=np.uint8, count=1)
        labels.append(label[0])

print(labels[:100])
print(len(labels))

gv = samples[70].reshape(NX, NY)

plt.gray()
plt.imshow(gv)
plt.show()