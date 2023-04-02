import numpy as np
import matplotlib.pyplot
from PIL import Image

NX = NY = 28
N = NX*NY

test_samples = []

# Load the test data from the file MNist_ttt4275/test_images.bin which is a binary file
# with 10000 images of size 28x28
with open('MNist_ttt4275/test_images.bin', 'rb') as f:
    for i in range(10000):
        test_samples.append(np.fromfile(f, dtype=np.uint8, count=28*28))

matrix_samples = []

print_sample = test_samples[0].reshape(NX, NY)
im = Image.fromarray(print_sample)
im.show()
