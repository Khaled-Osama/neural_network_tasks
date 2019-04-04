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
from tqdm import tqdm
'''
Loading data from file then shuffle them and split the data to training and testing set

'''

def load_data(all_data=False):

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



    X = df.iloc[:, :4].values

    Y = np.zeros((50, 3), dtype=np.int8)
    Y[:, 0] = 1

    X1_training, X1_testing, Y1_training, Y1_testing = train_test_split(X[:50, :], Y, test_size=0.4)

    Y = np.zeros((50, 3))
    Y[:,1] = 1
    X2_training, X2_testing, Y2_training, Y2_testing = train_test_split(X[50:100, :], Y, test_size=0.4)

    Y = np.zeros((50, 3))
    Y[:, 2] = 1
    X3_training, X3_testing, Y3_training, Y3_testing = train_test_split(X[100:150, :], Y, test_size=0.4)

    X_training = np.concatenate((X1_training, X2_training, X3_training), axis=0)
    X_testing = np.concatenate((X1_testing, X2_testing, X3_testing), axis=0)

    Y_training = np.concatenate((Y1_training, Y2_training, Y3_training), axis=0)
    Y_testing = np.concatenate((Y1_testing, Y2_testing, Y3_testing), axis=0)

    X_training, Y_training = shuffle(X_training, Y_training)
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

def initialize_weights(use_bias, num_of_hidden_layers, neurons):

    neurons.append(3)
    parameters = {}
    for i in range(num_of_hidden_layers + 1):
        W = None
        if i == 0:
            W = np.random.randn(neurons[0], 4)
        else:
            W = np.random.randn(neurons[i], neurons[i-1] )
        if use_bias:
            bias = np.random.randn(1, neurons[i])
        else:
            bias = np.zeros((1, neurons[i]))
        parameters['W'+str(i)] = W
        parameters['b' + str(i)] = bias

    return parameters

def activation_function(net, activation):
    if activation=='sigmoid':
        return 1.0 / (1.0 + np.exp(-net))
    else:
        return np.tanh(-net)


def derivative_activation(net, activation):
    if activation=='sigmoid':
        return activation_function(net, activation) * (1.0 - activation_function(net, activation))
    else:
        return (1.0 - np.tanh(net)) + (1.0 + np.tanh(net))


def forward_step(input, num_of_hidden_layers, parameters, activation):
    '''
    X -> dim (1,4)
    W -> dim(6,4)
    '''
    output_dic = {}
    nets_without_act = {}
    for i in range(num_of_hidden_layers):

        net = np.dot(input, parameters['W'+str(i)].T) + parameters['b'+str(i)]
        nets_without_act['Net' + str(i)] = net
        net = activation_function(net, activation)
        output_dic['Net'+str(i)] = net
        input = net

    net = np.dot(input, parameters['W' + str(num_of_hidden_layers)].T) + parameters['b' + str(num_of_hidden_layers)]
    nets_without_act['Net' + str(num_of_hidden_layers)] = net
    output_dic['Net' + str(num_of_hidden_layers)] = activation_function(net, activation)
    return output_dic, nets_without_act

def backward_step(parameters, nets, num_of_hidden_layers, Y_training, activation, nets_without_act):
    error ={}
    for i in range(num_of_hidden_layers, -1, -1):
        if i == num_of_hidden_layers:
            d = Y_training - nets['Net'+str(i)] * derivative_activation(nets_without_act['Net' + str(i)], activation)
        else:
            net = nets_without_act['Net' + str(i)]
            d = derivative_activation(net, activation) * np.dot(error['d' + str(i+1)], parameters['W'+str(i+1)])

        error['d' + str(i)] = d
    return error

def update_weights(parameters, errors, nets, alpha, number_hidden_layers, input, use_bias):

    for i in range(number_hidden_layers + 1):


        delta = np.dot(errors['d'+str(i)].T, input) * alpha
        parameters['W' + str(i)] = parameters['W' + str(i)] + delta
        if use_bias:
            parameters['b' + str(i)] = parameters['b' + str(i)] + (alpha * errors['d'+str(i)])
        input = nets['Net' + str(i)]

    return parameters

def train(X_training, Y_training, alpha, add_biase, num_of_hidden_layers, neurons, activation, epochs, MSE_threshold=0.5):

    parameters = initialize_weights(add_biase, num_of_hidden_layers, neurons)

    for _ in tqdm(range(epochs)):
        for j in range(len(X_training)):
            input = X_training[j, np.newaxis]
            forward_output, nets_without_act = forward_step(input=input,
                                          num_of_hidden_layers=num_of_hidden_layers,
                                          parameters=parameters,
                                          activation=activation)
            Y = Y_training[j, np.newaxis]

            backward_output = backward_step(parameters=parameters,
                                            nets = forward_output,
                                            num_of_hidden_layers=num_of_hidden_layers,
                                            Y_training=Y,
                                            activation=activation,
                                            nets_without_act=nets_without_act)

            parameters = update_weights(parameters=parameters,
                                        errors = backward_output,
                                        nets=forward_output,
                                        alpha=alpha,
                                        number_hidden_layers = num_of_hidden_layers,
                                        input = X_training[j,np.newaxis], use_bias=add_biase)

    return parameters

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
def get_testing_predictions(X_testing, parameters, activation, num_hidden_layers):
    '''
        X_Testing shape(60, 3)
        adding biase parameter
    '''

    layer = X_testing
    for i in range(num_hidden_layers + 1):
        layer = np.dot(layer, parameters['W' + str(i)].T)
        layer = activation_function(layer, activation)
    Y_pred = np.argmax(layer, axis=1)
    return Y_pred

######################################################################################################################


def calc_confusion_matrix(Y_pred, Y_testing):
    mat = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            k = 0
            for l in range(Y_testing.shape[0]):
                if Y_testing[l] == i and Y_pred[l] == j:
                    k += 1
                mat[i, j] = k
    return mat

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
    num_of_hidden_layers = int(hidden_layers_entry.get())
    neurons = neurons_entry.get().split()
    neurons = [int(x) for x in neurons]

    if(len(neurons) != num_of_hidden_layers):
        print("you must specify "+str(num_of_hidden_layers)+ "numbers of neurons")
        return

    X_training, X_testing, Y_training, Y_testing = load_data()

    add_biase = True if var.get() == 0 else False
    activation = 'sigmoid' if var1.get() == 0 else 'hyperbolic_tangent'

    parameters = train(X_training, Y_training,
              alpha=float(eta_entry.get()),
              add_biase=add_biase,
              num_of_hidden_layers=num_of_hidden_layers,
              neurons=neurons,
              activation=activation,
              epochs = int(epochs_entry.get()))

    Y_pred = get_testing_predictions(X_testing, parameters, activation, num_of_hidden_layers)
    matrix = calc_confusion_matrix(Y_pred=Y_pred, Y_testing=np.argmax(Y_testing, axis=1))
    print('Confusion matrix\n')
    print(matrix)
    accuracy = accuracy_score(np.argmax(Y_testing, axis=1), Y_pred)
    print('total all accuracy = ' + str(accuracy))

######################################################################################################################

root = tk.Tk()
#root.geometry("350x200")



f1_label = tk.Label(root, text='num o hidden layers', padx=10).grid(row=0)

f2_label = tk.Label(root, text='neurons of each layer', padx=10).grid(row=5)

epochs_label = tk.Label(root, text='epochs', padx=10).grid(row=10)


eta_label = tk.Label(root, text='learning Rate', padx=10).grid(row=15)


#MSE_label = tk.Label(root, text='MSE Threshold', padx=10).grid(row=20)

bias_label = tk.Label(root, text="add biase", padx=10).grid(row=25)

activation_label = tk.Label(root, text='activation', padx=10).grid(row=30)


hidden_layers_entry = Entry(root)
neurons_entry = Entry(root)


eta_entry = Entry(root)
epochs_entry = Entry(root)
#MSE_entry = Entry(root)

hidden_layers_entry.grid(row=0, column=4)
neurons_entry.grid(row=5, column=4)

epochs_entry.grid(row=10, column=4)

eta_entry.grid(row=15, column=4)
#MSE_entry.grid(rows=20, column=4)

var = tk.IntVar()
var.set(0)

var1 = tk.IntVar()
var1.set(0)

Radiobutton(root, text="yes", variable=var, value=0).grid(row=25, column=1)
Radiobutton(root, text="no", variable=var, value=1).grid(row=25, column=4)

Radiobutton(root, text="sigmoid", variable=var1, value=0).grid(row=30, column=1)
Radiobutton(root, text="hyperbolic Tangent", variable=var1, value=1).grid(row=30, column=4)

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