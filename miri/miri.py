import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def plot():
    plt.scatter(epochar, tr_mlp, c = "red")
    plt.scatter(epochar, tr_sgd, c = "blue")
    plt.legend(["SGD", "Adam"], loc = "lower right")
    plt.xlabel("Epoch")
    plt.ylabel("Average Squared Error")
    plt.show()
    plt.savefig("plot.png")

# get training data
def getdata(file, dir):
    list = []
    with open(file, "r") as files:
        for line in files:
            with open(dir + "/" + line.strip("\n"), "r") as f:
                for l in f:
                    list.append([float(i) for i in l.split(",")])
    return list

def splitten(input):
    t = 11 # change this in terms of dev split
    tr = [] # train
    de = [] # dev
    for i in range (len(input)):
        if i % t == 0:
            de.append(input[i])
        else:
            tr.append(input[i])
    return tr, de

# Train
Xtr = getdata("small-x", "small")
Ytr = pd.read_csv("small-y", header=None).values

# Dev data split
Xtr, Xte = splitten(Xtr)
Ytr, Yte = splitten(Ytr)

tr_sgd = []
te_sgd = []
tr_mlp = []
te_mlp = []
epochar = []

# Converts to np array before combining
Xtr = np.array(Xtr, dtype=float)
Xte = np.array(Xte, dtype=float)
Ytr = np.array(Ytr, dtype=int).flatten()
Yte = np.array(Yte, dtype=int).flatten()

# Combine data for ease of use
X = {'train': Xtr, 'test': Xte}
Y = {'train': Ytr, 'test': Yte}

# Hyper parameters

# STANDARD
# Best(8) epoch: 100, layers: 200, rate: 0.005
# Best(9) epoch: 200, layers: 500, rate: 0.001
# Best(10) epoch: 200, layers: 200, rate: 0.001
# At this point epoch of 200 is used for tuning the other two params

# Layers - 11
# Best(ep = 100) layers: 300, rate: 0.001
# Best(ep = 200) layers: 200, rate: 0.001 
# epoch = [100, 200, 300, 400, 500, 1000, 10000]
# layers = [100, 200, 300]
# rates = [0.1, 0.001, 0.005, 0.0001]

# Hyperparameters - MLP
lr = 0.001    # learning rate
# ep = 200      # epoch
bs = 100      # batch size
l = 200       # layers

# Hyperparameters - Forest
estimator = [50, 100, 150, 200]
maxdepth = [5, 10, 15, 20]
leaves = [1, 2, 3, 4, 5, 10, 20]

grid = {'solver': ['adam'],
        'max_iter': [100, 200, 500],
        'batch_size': [50, 100],
        'hidden_layer_sizes': [(l, l, l, l), (l, l)],
        'learning_rate_init': [0.001]}

print "Training...\n"
ti = time.time()


# Final Accuracy on sets (HYPERPARAMETER TUNING)
# lm = LinearRegression()
# for e in epoch:
#     for l in layers:
#         for r in rates:
#             mlp = MLPClassifier(solver='adam', batch_size = bs, hidden_layer_sizes=(l, l, l, l), learning_rate_init=l, max_iter=ep)
#             mlp.fit(X['train'], Y['train'])
#             print "epoch: " + str(e) + ", layers: " + str(l) + ", rate: " + str(r)

# Tuned hyperparameters
for ep in range (1, 201):
    sgd = MLPClassifier(solver='sgd', max_iter=ep)
    sgd.fit(X['train'], Y['train'])
    mlp = MLPClassifier(solver='adam', max_iter=ep)
    mlp.fit(X['train'], Y['train'])
    # tf = time.time()
    print "epoch: " + str(ep) + "\n"
    print "SGD Training loss: " + str(mean_squared_error(Y['train'], sgd.predict(X['train'])))
    print "SGD Test loss: " + str(mean_squared_error(Y['test'], sgd.predict(X['test']))) + "\n"
    print "ADAM Training loss: " + str(mean_squared_error(Y['train'], mlp.predict(X['train'])))
    print "ADAM Test loss: " + str(mean_squared_error(Y['test'], mlp.predict(X['test']))) + "\n"
    # tr_sgd.append(mean_squared_error(Y['train'], sgd.predict(X['train'])))
    # te_sgd.append(mean_squared_error(Y['test'], sgd.predict(X['test'])))
    # tr_mlp.append(mean_squared_error(Y['train'], mlp.predict(X['train'])))
    # te_mlp.append(mean_squared_error(Y['test'], mlp.predict(X['test'])))
    tr_sgd.append(sgd.score(X['train'], Y['train']))
    te_sgd.append(sgd.score(X['test'], Y['test']))
    tr_mlp.append(mlp.score(X['train'], Y['train']))
    te_mlp.append(mlp.score(X['test'], Y['test']))
    epochar.append(ep)

plot()
# mlp2 = MLPClassifier(activation='tanh' ,solver='adam', batch_size = bs, hidden_layer_sizes=(l, l, l, l), learning_rate_init=l, max_iter=ep)
# mlp2.fit(X['train'], Y['train'])
# gbr = RandomForestRegressor(n_estimators=6000)
# gbr.fit(X['train'], Y['train'])
# ran = RandomForestRegressor()
# ran.fit(X['train'], Y['train'])
# bag = BaggingRegressor()
# bag.fit(X['train'], Y['train'])

# for nest in estimator:
#     for depth in maxdepth:
#         for leaf in leaves:
#             For = RandomForestRegressor(n_estimators=nest, max_depth=depth, min_samples_leaf=leaf)
#             For.fit(X['train'], Y['train'])
#             print "n_est: " + str(nest) + ", max depth: " + str(depth) + ", min_leaf: " + str(leaf)
#             print "FOR Training loss: " + str(mean_squared_error(Y['train'], For.predict(X['train'])))
#             print "FOR Test loss: " + str(mean_squared_error(Y['test'], For.predict(X['test']))) + "\n"

# Tuning with library
# clf = GridSearchCV(MLPClassifier(), grid, scoring='neg_mean_squared_error', verbose=10)
# clf.fit(X['train'], Y['train'])

# SVR params doesnt seem to have effect
# lin = LinearSVR(C=1.0, dual=True, epsilon=0.0, fit_intercept=True,
#       intercept_scaling=1.0, loss='epsilon_insensitive', max_iter=1000,
#       random_state=0, tol=0.0001, verbose=0)
# svr = SVR(kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
# sgd = SGDClassifier(loss="hinge", penalty="l2")
# Lasso and poly
# las = Lasso(alpha = 0.1)
# pol = PolynomialFeatures(degree=8)
# com = Pipeline([('poly', PolynomialFeatures(degree=8)),
#                  ('linear', LinearRegression(fit_intercept=False))])

# svr.fit(X['train'], Y['train'])
# las.fit(X['train'], Y['train'])
# sgd.fit(X['train'], Y['train'])
tf = time.time()

# print "TAN Training loss: " + str(mean_squared_error(Y['train'], mlp2.predict(X['train'])))
# print "TAN Test loss: " + str(mean_squared_error(Y['test'], mlp2.predict(X['test']))) + "\n"
# print "BAG Training loss: " + str(mean_squared_error(Y['train'], bag.predict(X['train'])))
# print "BAG Test loss: " + str(mean_squared_error(Y['test'], bag.predict(X['test']))) + "\n"
# print "RAN Training loss: " + str(mean_squared_error(Y['train'], ran.predict(X['train'])))
# print "RAN Test loss: " + str(mean_squared_error(Y['test'], ran.predict(X['test']))) + "\n"
# print "LAS Training loss: " + str(mean_squared_error(Y['train'], las.predict(X['train'])))
# print "LAS Test loss: " + str(mean_squared_error(Y['test'], las.predict(X['test']))) + "\n"
# print "GBR Training loss: " + str(mean_squared_error(Y['train'], gbr.predict(X['train'])))
# print "GBR Test loss: " + str(mean_squared_error(Y['test'], gbr.predict(X['test']))) + "\n"

print "Time Taken: " + "%0.2f" % (tf - ti)

# Prediction and output
testx = np.array(getdata("test-x", "test"), dtype=int)
with open("31.8.test-yhat", 'w') as output:
    for i in mlp.predict(testx):
        print >> output, i