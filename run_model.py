import torch as ts
import torch.nn as nn

ts.manual_seed(1)

import numpy as np
import random
import time

from mlxtend.data import loadlocal_mnist

def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    ts.manual_seed(seed)
    ts.cuda.manual_seed_all(seed)
    # For atomic operations there is currently 
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    ts.backends.cudnn.deterministic = True
    ts.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)

enforce_reproducibility()

device = ts.device("cpu")
if ts.cuda.is_available():
    print("has cuda")
    device = ts.device("cuda")

"""
Converting things into tensors
Converting things into tensors
Converting things into tensors
"""

def x2tensor(x_in):
    return ts.tensor([float(x) for x in x_in]).to(device)

def y2tensor(y_in):
    return ts.LongTensor([y_in]).to(device)



"""
Training the model
Training the model
Training the model
"""

def comp_time(t0):
    used_time = round(time.time() - t0,2)
    measure = "seconds"
    if used_time >= 60.0:
        used_time /= 60.0
        used_time = round(used_time,2)
        measure = "minutes"
    if used_time >= 60.0:
        used_time /= 60.0
        used_time = round(used_time,2)
        measure = "hours"
    return str(used_time) + " " + measure

def train(model,lrate,n_epochs,optim_fun):
    X, y = loadlocal_mnist("mnist_data/train-images.txt","mnist_data/train-labels.txt")
    
    model.to(device)

    loss_fun = nn.NLLLoss()
    optimizer = optim_fun(model.parameters(),lr=lrate)

    for epoch in range(n_epochs):
        avg_loss = 0
        start_time = time.time()
        for (img,label) in zip(X,y):
            model.zero_grad()

            x = x2tensor(img)

            target = y2tensor(label)

            logits = model(x)

            loss = loss_fun(logits,target)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        print("loss[" + str(epoch) + "] = " + str(avg_loss / X.shape[0]))
        print("------------------------------------took " + comp_time(start_time))

"""
Testing the model
Testing the model
Testing the model
"""

def save_erate(mname,neps,res):
    fname = mname + ".erates"
    with open(fname,"a") as f:
        f.write("[" + str(neps) + "]" + res + "\n")
        print("e-rate saved in " + fname)

def run_model(model,sample):
    logits = None
    with ts.no_grad():
        logits = model(x2tensor(sample))
    return logits.argmax().item()

def test(model_name,model,n_epochs):
    X_test, y_test = loadlocal_mnist("mnist_data/test-images.txt","mnist_data/test-labels.txt")

    correct = 0
    misses = 0

    N = X_test.shape[0]

    for (img,label) in zip(X_test,y_test):
        res = run_model(model,img)
        if res == label:
            correct += 1
        else:
            misses += 1
    
    print(str(correct) + "/" + str(N))
    print("e-rate = " + str(misses / N))
    save_erate(model_name,n_epochs,str(misses / N))
