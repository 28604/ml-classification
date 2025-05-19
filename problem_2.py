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
            data_T = np.vstack([data_T, i])
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
    with open('./outputs/result_2.csv', 'w') as f:
        for i in range(len(y_pred)):
            f.write(f"{y_pred[i]}\n")

#-------------------------------------------------------------------------

class MNISTmulti:
    def __init__(self, M=784, K=10):
        self.M = M
        self.K = K
        self.W = None  # W shape: (M + 1, K)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500) # prevent overflow
        return 1 / (1 + np.exp(-z))

    def compute_Phi(self, X, d=1):
        N = X.shape[0]
        Phi = X**d
        # Phi = self.sigmoid(X)
        Phi = np.hstack([np.ones((N, 1)), Phi])  # add bias
        return Phi  # (N, M+1)
    
    # activation function
    def softmax(self, A):
        ''' calculate P(Ci | x, wi), returns a N*K matrix with all information'''
        A = np.clip(A, -500, 500) # prevent overflow
        expA = np.exp(A)
        return expA / np.sum(expA, axis=1, keepdims=True)  # shape: (N, K)

    def one_hot(self, t, K):
        ''' for each row, encode numbers in one hot format, returns N*K matrix '''
        N = t.shape[0] 
        one_hot_matrix = np.zeros((N, K))
        one_hot_matrix[np.arange(N), t.flatten()] = 1
        return one_hot_matrix

    def gradient_descent(self, X, t, d=1, max_iter=1000, eta=0.1, epsilon=1e-4):
        Phi = self.compute_Phi(X, d)                    # (N, M+1)
        self.W = np.random.randn(Phi.shape[1], self.K)  # (M+1, K)
        T = self.one_hot(t, self.K)                     # (N, K)

        for i in range(max_iter):
            A = Phi @ self.W                       # (N, K)
            Y = self.softmax(A)                    # (N, K)
            grad_W = Phi.T @ (Y - T)               # (M+1, K)

            # if np.linalg.norm(grad_W) < epsilon:
            #     break
            self.W -= eta * grad_W
            eta *= 0.99

            if i % 100 == 0:
                loss = -np.sum(np.sum(T * np.log(Y + 1e-8), axis=1))
                print(f"iter {i}, loss: {loss:.4f}, gradient norm: {np.linalg.norm(grad_W):.4f}")

    def predict(self, X):
        Phi = self.compute_Phi(X)
        A = Phi @ self.W
        Y = self.softmax(A)
        return np.argmax(Y, axis=1)  # returns predicted class labels


def s_fold_cross_validation(X, t, S=5):
    N = X.shape[0]
    indices = np.random.permutation(N)
    folds = np.array_split(indices, S)

    ds = np.arange(30, 31, 1)
    avg_accs = []
    for d in ds:
        accs = []
        for i in range(S):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(S) if j != i])
            X_train, t_train = X[train_idx], t[train_idx]
            X_val, t_val = X[val_idx], t[val_idx]

            model = MNISTmulti(784, 10)
            model.gradient_descent(X_train, t_train, d=d, max_iter=800, eta=0.05, epsilon=1000)

            y_pred = model.predict(X_val)
            acc = np.mean(y_pred.reshape(-1, 1) == t_val)
            accs.append(acc)

            # print(f"fold {i+1}/{S}, accuracy: {acc * 100:.2f}%")

        avg_acc = np.mean(accs)
        avg_accs.append(avg_acc)
        # print(f"\naverage accuracy: {avg_acc * 100:.2f}%")
        print(f"\npower={d}, average accuracy: {avg_acc:.4f}")

    plt.plot(ds, avg_accs, 'o')
    plt.title('Accuracy of binary classification')
    plt.xlabel('power of basis function')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    X_train, t_train = read_train_data()
    X_train = X_train / 255.0
    # s_fold_cross_validation(X_train, t_train, S=5)

    model = MNISTmulti()
    model.gradient_descent(X_train, t_train, d=30, max_iter=1000, eta=0.05, epsilon=1000)

    X_test = read_test_data()
    X_test = X_test / 255.0
    y_pred = model.predict(X_test)
    save_result(y_pred)
