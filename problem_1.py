import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from PIL import Image
import os

def read_train_data():
    data_X = np.zeros((0, 28*28))
    data_T = np.zeros((0, 1), dtype=int)
    for i in range(10):
        for j in range(1000):
            img = np.array(Image.open(f"train/{i}/{j}.jpg"))
            data_X = np.vstack([data_X, img.reshape((1, 28*28))])
            data_T = np.vstack([data_T, i%2])
    return data_X, data_T

def read_test_data():
    data_X = np.zeros((0, 28*28))
    data_T = np.zeros((0, 1), dtype=int)
    for i in range(2000):
        img = np.array(Image.open(f"test/{i}.jpg"))
        data_X = np.vstack([data_X, img.reshape((1, 28*28))])
        data_T = np.vstack([data_T, i%2])
    return data_X

def save_result(y_pred):
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    with open('./outputs/result_1.csv', 'w') as f:
        for i in range(len(y_pred)):
            f.write(f"{y_pred[i, 0]}\n")

#-------------------------------------------------------------------------

class MNISTbinary:
    def __init__(self, M):
        self.M = M
        self.w = None

    # activation function
    def sigmoid(self, z):
        z = np.clip(z, -500, 500) # prevent overflow
        return 1 / (1 + np.exp(-z))

    # design matrix
    def compute_Phi(self, X, d=2):
        ''' Phi(N * (1 + M)), design matrix '''
        N = X.shape[0]
        Phi = X**d
        Phi = np.hstack([np.ones((N, 1)), Phi]) # add bias
        return Phi # (N, M+1)

    # training model
    def gradient_descent(self, X, t, d=2, max_iter=300, eta=0.01, epsilon=1000):
        Phi = self.compute_Phi(X, d)
        self.w = np.random.randn(Phi.shape[1], 1)
        for i in range(max_iter):
            y = self.sigmoid(Phi @ self.w)
            gradient = Phi.T @ (y - t)
            if np.linalg.norm(gradient) < epsilon:
                break
            self.w = self.w - eta * gradient
            loss = -np.mean(t * np.log(y + 1e-8) + (1 - t) * np.log(1 - y + 1e-8))
            if i % 100 == 0:
                eta = eta * 0.85
                print(f"iter {i}, loss: {loss:.4f}, gradient norm: {np.linalg.norm(gradient):.4f}")

    # predict numbers 
    def predict(self, X):
        Phi = self.compute_Phi(X)
        probs = self.sigmoid(Phi @ self.w)
        return (probs >= 0.5).astype(int)
    
    def accuracy(self, X, t):
        y_pred = self.predict(X)
        return np.mean(y_pred == t)
    
#-------------------------------------------------------------------------
def s_fold_cross_validation(X, t, S=5):
    N = X.shape[0]
    indices = np.random.permutation(N)
    folds = np.array_split(indices, S)
    
    Ms = np.arange(784, 785, 1)
    ds = np.arange(1, 2, 1)
    avg_accs = []
    for M in Ms:
        for d in ds:
            accs = []
            for i in range(S):
                val_idx = folds[i] # fold i is validation set
                train_idx = np.hstack([folds[j] for j in range(S) if j != i]) # folds - fold[i] is training set

                X_train, t_train = X[train_idx], t[train_idx]
                X_val, t_val = X[val_idx], t[val_idx]

                model = MNISTbinary(M=M)
                model.gradient_descent(X_train, t_train, d=d, max_iter=5000, eta=0.3, epsilon=300)

                acc = model.accuracy(X_val, t_val)
                accs.append(acc)

                print(f"fold {i+1}/{S}, accuracy: {acc * 100:.2f}%")

            avg_acc = np.mean(accs)
            avg_accs.append(avg_acc)
            print(f"average accuracy: {avg_acc * 100:.2f}%")
            # print(f"M = {M}, average accuracy: {avg_acc:.4f}")
            # print(f"power = {d}, average accuracy: {avg_acc:.4f}")
    
    # plt.plot(ds, avg_accs, 'o')
    # plt.title('Accuracy of binary classification (M=784)')
    # plt.xlabel('power of basis function')
    # plt.ylabel('Accuracy')
    # plt.show()

#-------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, t_train = read_train_data()
    X_train = X_train / 255.0
    t_train = t_train.astype(float)
    # s_fold_cross_validation(X_train, t_train, S=5)
    
    X_test = read_test_data() 
    X_test = X_test / 255.0 
    M = 784
    d = 2
    model = MNISTbinary(M=M)
    model.gradient_descent(X_train, t_train, d=d, max_iter=5000, eta=0.3, epsilon=300)
    model.predict(X_test)
    y_test_pred = model.predict(X_test)
    save_result(y_test_pred)
