# ml-classification
Binary and multi-class classification algorithm using linear models coded from scratch.

### Problem: classify MNIST dataset using binary classification (odd / even) and multi-class classification (0-9)

<img src="https://github.com/28604/ml-classification/blob/main/img/MNIST.png" width="600" alt="An image of MNIST dataset">

### Description
The training dataset contains 10,000 images (28x28 grayscale) with 1,000 samples per digit. You should create feature vectors based on the input coordinates as the model’s inputs and train your binary classification and multi-class classification model to classify the training data. Predict the class for the 2,000 images in the testing dataset and save your predictions as a .csv file. For each approach, the **accuray of your predictions on the testing dataset must be more than 85%**; otherwise, you will fail the correctness check.

### Grading policy
If the testing accuracy of your prediction is greater than 85%, you will receive full credits.

### Classification method
* **Binary Classification** (sigmoid activation function)

  <img src="https://github.com/28604/ml-classification/blob/main/img/binary%20classification.png" width="400" alt="An image of binary classification pipeline"> 
  
* **Multi-class Classification** (softmax activation function)

  <img src="https://github.com/28604/ml-classification/blob/main/img/multi-class%20classification.png" width="400" alt="An image of mult-class classification pipeline"> 

### Other Details
* K-fold cross validation
* Gradient descent for less computation than Newton-Raphson method
* One-hot encoding for multi-class classification

### Challenge
* Find the best basis function to make the features as linearly separable as possible
