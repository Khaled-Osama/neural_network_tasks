import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import Label, Entry, Radiobutton, Button
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import seaborn as sn
'''
Loading data from file then shuffle them and split the data to training and testing set

'''

def load_data(features=None, classes=None, all_data=False):

    file_name = "IrisData.txt"
    df = pd.read_csv(file_name)


    if all_data:

        Map = {"Iris-setosa":0,
      "Iris-versicolor":1,
      "Iris-virginica":2}

        X = df.iloc[:, 0:4].values
        Y = df.iloc[:, 4].values
        Y = [Map[x] for x in Y]
        Y = np.asarray(Y, dtype=np.float32)
        Y = Y[:, np.newaxis]
        return X, Y


    c1, c2 = classes
    x1, x2 = features

    indices1 = np.arange(50*c1, (c1+1)*50)
    indices2 = np.arange(50*c2, 50*(c2+1))

    X = df.iloc[:, [x1, x2]].values

    Y = np.ones((50, 1))
    X1_training, X1_testing, Y1_training, Y1_testing = train_test_split(X[indices1, :], Y, test_size=0.4, random_state=0)
    Y[:,:] = -1
    X2_training, X2_testing, Y2_training, Y2_testing = train_test_split(X[indices2, :], Y, test_size=0.4, random_state=0)

    X_training = np.concatenate((X1_training, X2_training), axis=0)
    X_testing = np.concatenate((X1_testing, X2_testing), axis=0)

    Y_training = np.concatenate((Y1_training, Y2_training), axis=0)
    Y_testing = np.concatenate((Y1_testing, Y2_testing), axis=0)

    X_training, Y_training = shuffle(X_training, Y_training, random_state=0)
    X_testing, Y_testing = shuffle(X_testing, Y_testing)

    return X_training, X_testing, Y_training, Y_testing

##################################################################################################################################
'''
plot the data
'''
def plot(X1, X2, fig_num):
    plt.figure('fig' + str(fig_num))
    plt.scatter(X1[:50], X2[:50], color='red')
    plt.scatter(X1[50:100], X2[50:100], color='green')
    plt.scatter(X1[100:150], X2[100:150], color='blue')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

##################################################################################################################################
'''


'''
def calc_LMSE(W, X_training, Y_training):
    Y_pred = np.dot(X_training, W)
    loss = np.subtract(Y_training, Y_pred)
    loss_square = np.square(loss)
    return np.sum( loss_square )/(2*Y_training.shape[0])

##################################################################################################################################
'''
train function initialize weights randomly and update the weights for in each missclassification
after each epoch calculate the Mean Square Error over all training examples.
terminate when MSE smaller than some threshold

'''

def train(X_training, Y_training, alpha, add_biase, MSE_threshold):
    W = np.random.randn(3, 1)
    if add_biase == True:
        X_training =  np.insert(X_training, 0, 1, axis=1)
    else:
        tmp = X_training
        W[0,0] = 0
        X_training = np.zeros((60, 3))
        X_training [:, 1:3] = tmp

    error = 50

    while error > MSE_threshold:

        for x in range(60):
            y_pred = np.dot(X_training[x, :], W)
            L = Y_training[x, 0] - y_pred #Loss
            K = X_training[x, :]
            K = K[:,np.newaxis]
            W = W + alpha*(L * K)
        error = calc_LMSE(W, X_training, Y_training)
        print(error)
    return W

######################################################################################################################
'''
activation function that's used in testing only since the adalinear activation function is linear
signum used while predicting to map our output to it's corresponding class.
'''
def signum(X):
    X[X>0] = 1
    X[X<0] = -1
    return X

######################################################################################################################
'''
predict the classes of the testing set
'''
def get_testing_predictions(X_testing, Y_testing, W):
    '''
        X_Testing shape(40, 2)
        adding biase parameter
    '''
    tmp = np.ones((40, 3))
    tmp[:, 1:3] = X_testing
    Y_pred = np.dot(tmp, W)
    Y_pred = signum(Y_pred)
    return Y_pred

######################################################################################################################
'''
plot the line which comes from adalinear algorithm hoping that it splitting our linear separable data.
'''
def plot_line(X_training, Y_training, W):
    X_training = np.insert(X_training, 0, 1, axis=1)

    minn = min(X_training[:, 1])
    maxx = max(X_training[:, 1])
    x = [minn, maxx]
    y = [(-W[0, 0] - W[1, 0] * minn) / W[2, 0], (-W[0, 0] - W[1, 0] * maxx) / W[2, 0]]
    l1 = []
    l2 = []
    l3 = []
    l4 = []

    for i in range(40):
        if Y_training[i] == 1:
            l1.append(X_training[i, 1])
            l2.append(X_training[i, 2])
        else:
            l3.append(X_training[i, 1])
            l4.append(X_training[i, 2])

    plt.plot(x, y)
    plt.scatter(l1, l2, color='green')
    plt.scatter(l3, l4, color='blue')

    plt.show()
#######################################################################################################################
'''
this function called to start the whole training operations
'''

def start_learning():
    features = [int(f1_entry.get()), int(f2_entry.get())]
    classes = [int(c1_entry.get()), int(c2_entry.get())]
    X_training, X_testing, Y_training, Y_testing = load_data(features=features, classes=classes)

    add_biase = True if var.get() == 0 else False

    W = train(X_training, Y_training,
              alpha=float(eta_entry.get()),
              add_biase=add_biase,
              MSE_threshold=float(MSE_entry.get()))

    Y_pred = get_testing_predictions(X_testing, Y_testing, W)
    accuracy = accuracy_score(Y_testing, Y_pred)
    print('total all accuracy = ' + str(accuracy))
    # plot_confusion_matrix(Y_testing, Y_pred)

    plot_line(X_testing, Y_testing, W)
######################################################################################################################

root = tk.Tk()
#root.geometry("350x200")



f1_label = tk.Label(root, text='Feature 1', padx=10).grid(row=0)

f2_label = tk.Label(root, text='Feature 2', padx=10).grid(row=5)

c1_label = tk.Label(root, text='Class 1', padx=10).grid(row=10)

c2_label = tk.Label(root, text='Class 2', padx=10).grid(row=15)

eta_label = tk.Label(root, text='learning Rate', padx=10).grid(row=20)


MSE_label = tk.Label(root, text='MSE Threshold', padx=10).grid(row=25)

bias_label = tk.Label(root, text="add biase", padx=10).grid(row=30)

f1_entry = Entry(root)
f2_entry = Entry(root)

c1_entry = Entry(root)
c2_entry = Entry(root)

eta_entry = Entry(root)
epochs_entry = Entry(root)
MSE_entry = Entry(root)

f1_entry.grid(row=0, column=4)
f2_entry.grid(row=5, column=4)

c1_entry.grid(row=10, column=4)
c2_entry.grid(row=15, column=4)

eta_entry.grid(row=20, column=4)
MSE_entry.grid(rows=25, column=4)

var = tk.IntVar()
var.set(0)


Radiobutton(root, text="yes", variable=var, value=0).grid(row=30, column=1)
Radiobutton(root, text="no", variable=var, value=1).grid(row=30, column=4)

submit = tk.Button(root,
                   text="train",
                   fg="red",
                   command=start_learning).grid(row=50, column=3)



root.mainloop()

######################################################################################################################



######################################################################################################################



'''X, Y = load_data(all_data=True)
plot(X[:,0], X[:,1], 1)
plot(X[:,0], X[:,2], 2)
plot(X[:,0], X[:,3], 3)
plot(X[:,1], X[:,2], 4)
plot(X[:,1], X[:,3], 5)
plot(X[:,2], X[:,3], 6)'''