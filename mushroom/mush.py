import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# get training data
def getdata(file, dir):
    list = []
    with open(file, "r") as files:
        for line in files:
            with open(dir + "/" + line.strip("\n"), "r") as f:
                for l in f:
                    list.append([int(i) for i in l.split()])
    return list

def plot():
    plt.plot(epochar, tr_sgd, color = "red")
    plt.plot(epochar, tr_mlp, color = "blue")
    plt.legend(["SGD", "Adam"], loc = "lower right")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()
    plt.savefig("plot.png")

# Train
Xtr = getdata("train-x", "train")
Ytr = pd.read_csv("train-y", header=None).values

# Dev/test
Xte = getdata("development-x", "development")
Yte = pd.read_csv("development-y", header=None).values

# Converts to np array before combining
Xtr = np.array(Xtr, dtype=int)
Xte = np.array(Xte, dtype=int)
Ytr = np.array(Ytr, dtype=int).flatten()
Yte = np.array(Yte, dtype=int).flatten()

# Combine data for ease of use
X = {'train': Xtr, 'test': Xte}
Y = {'train': Ytr, 'test': Yte}

tr_sgd = []
te_sgd = []
tr_mlp = []
te_mlp = []
epochar = []

# Hyper parameters
# epoch = [10, 25, 50]
# layers = [100, 200, 500]
# rates = [0.001, 0.005, 0.1]
lr = 0.001    # learning rate
# ep = 50       # epoch
bs = 50       # batch size

print "Training..."
ti = time.time()

# Final Accuracy on sets (tuned)
for ep in range (1, 101):
    sgd = MLPClassifier(solver='sgd', max_iter=ep)
    sgd.fit(X['train'], Y['train'])
    mlp = MLPClassifier(solver='adam', max_iter=ep)
    mlp.fit(X['train'], Y['train'])
    # tf = time.time()
    print "epoch: " + str(ep) + "\n"
    print "SGD Training Accuracy: " + "%0.1f" % (sgd.score(X['train'], Y['train']) * 100) + "%"
    print "SGD Test Accuracy: " + "%0.1f" % (sgd.score(X['test'], Y['test']) * 100) + "%"
    print ""
    print "ADAM Training Accuracy: " + "%0.1f" % (mlp.score(X['train'], Y['train']) * 100) + "%"
    print "ADAM Test Accuracy: " + "%0.1f" % (mlp.score(X['test'], Y['test']) * 100) + "%"
    tr_sgd.append(sgd.score(X['train'], Y['train']) * 100)
    te_sgd.append(sgd.score(X['test'], Y['test']) * 100)
    tr_mlp.append(mlp.score(X['train'], Y['train']) * 100)
    te_mlp.append(mlp.score(X['test'], Y['test']) * 100)
    epochar.append(ep)

plot()


# print "Time Taken: " + "%0.2f" % (tf - ti)

# # Prediction and output
# testx = np.array(getdata("test-x", "test"), dtype=int)
# with open("31.7.test-yhat", 'w') as output:
#     for i in mlp.predict(testx):
#         print >> output, i