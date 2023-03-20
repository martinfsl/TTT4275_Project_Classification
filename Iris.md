# This is a file used to brainstorm / draft ideas to solve the iris task

1. 
Divide the features into a feature vector x:
    x = [sepal length, sepal width, petal length, petal width]
Multiplying this with a weighting matrix W. Will also have an offset vector w0.

Output vector:
    g = Wx + w0
This is used both to train the weights in W and w0, and later to classify the input.
