import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# FASHIONMNIST

TRAIN_SIZE, VAL_SIZE = 50000, 10000
dataset_prefix = "fashionmnist" 
num_classes = 10

train_data = np.load("./hw5_data/{}_train.npy".format(dataset_prefix))
test_data = np.load("./hw5_data/{}_test.npy".format(dataset_prefix))

# Note: I am unable to use sklearn at the moment due to package inconsistencies
# Split train data to train/val/test
train_images = train_data[:TRAIN_SIZE, :]
val_images = train_data[TRAIN_SIZE:, :]

test_images = test_data.reshape(-1, 1, 28, 28)



def imshow(image, title=None):
    fig, ax = plt.subplots(1, figsize=(2,2))
    ax.imshow(image.squeeze(0)*255, cmap='gray')
    if title is not None:
        plt.title(title)

# for i in range(10):
#     imshow(train_images[i], title='Image')

# PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

# CNN ARCHITECTURE: Conv2d(,,5) -> MaxPool2d(2,2) -> ReLU -> Conv2d(,,5) -> MaxPool2d(2,2) -> Linear(,) -> ReLu 
# -> Linear(, 10) -> Softmax

class Net(nn.Module):
    def __init__(self, S=1, P=0, conv1=(1, 10, 5), pool=2, pool_S=1, conv2=(10, 10, 5), fc1=20, fc2=10, drop=0.5, batch=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(conv1[0], conv1[1], conv1[2], stride=S, padding=P) # output dim=(BATCH_SIZE x c1 x (24+2P) x (24+2P))
        self.pool = nn.MaxPool2d(pool, stride=pool_S) # output dim=(BATCH_SIZE x c1 x (12+P) x (12+P))
        self.conv2 = nn.Conv2d(conv2[0], conv2[1], conv2[2], stride=1, padding=P) # output dim=(BATCH_SIZE x c2 x (8+3P) x (8+3P))
        self.l = conv2[1] * (4+int(1.5*P)) * (4+int(1.5*P))
        self.fc1 = nn.Linear(self.l, fc1, bias=True) 
        self.fc2 = nn.Linear(fc1, num_classes, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop)
        self.bn1 = nn.BatchNorm1d(num_features=fc1)
        self.batch = batch
        print(self.batch)
        

    def forward(self, x):
        x = F.relu(self.pool(self.dropout(self.conv1(x)))) # dim=(BATCH_SIZE x c1 x (12+P) x (12+P))
        x = self.pool(self.conv2(x))
        x = x.view(self.batch, self.l)
        x = self.fc1(x) 
        if self.batch > 1:
            x = self.bn1(x)
        x = self.softmax(self.fc2(F.relu(x)))
        
        return x


def train(S, P, conv1, pool, pool_S, conv2, fc1, drop, opt, lr, mom, batch):
    # NN INITIALIZATION
    net = Net(S=S, P=P, conv1=conv1, pool=pool, pool_S=pool_S, conv2=conv2, fc1=fc1, drop=drop, batch=batch)
        
    criterion = nn.CrossEntropyLoss()
    
    if opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom)
    elif opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.999))
    elif opt == 'nesterov':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom, nesterov=True)
    
    
    trainloader = torch.utils.data.DataLoader(train_images, batch_size=batch,
                                              shuffle=True, num_workers=2)
    epsilon = 0.002
    losses = []
    vals = []
    for epoch in range(10):  # loop over the dataset multiple times 
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[:, :-1]
            labels = data[:, -1]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if inputs.shape[0] == batch:
                outputs = net(inputs.reshape(batch, 1, 28, 28).float())
                outputs = outputs.reshape(batch, -1)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % (50000//(10*batch)) == ((50000//(10*batch))-1):    # print every 1/10 epoch
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (50000//(10*batch))))
                losses.append(running_loss/(50000//(10*batch)))
                running_loss = 0.0
                
                # val accuracy
                valloader = torch.utils.data.DataLoader(val_images, batch_size=batch,
                                                          shuffle=True, num_workers=2)
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in valloader:
                        images, labels = data[:, :-1], data[:, -1]
                        if images.shape == (batch, 784):
                            outputs = net(images.reshape(batch, 1, 28, 28).float())
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))
                vals.append(100 * correct / total)
                

        if losses[-9] - losses[-1] < epsilon:
            break

    print('Finished Training')
    return net, losses, vals


# Perform random search to tune hyperparameters; more efficient than grid search since it allows for identifying important
# parameters more quickly, and avoids prioritizing less important ones (Stanford CS231)

def random_search(num_iter=1):
    # S = 1 as good practice and simplified dimensional constraints
    # pool = 2, pool_S=2 as convention according to cs231
    # fc2 = 10 (num_labels) 
    pad = np.random.randint(1, 2, size=num_iter)*2 # since filter kernel is 5x5,P in {0,2,4}
    c1 = 2**np.random.randint(9, 10, size=num_iter) # conv1 in [32, 512], kernel is 5x5
    c2 = 2**np.random.randint(9, 10, size=num_iter) # conv2 in [32, 512], kernel is 5x5
    fc1 = 2**np.random.randint(9, 10, size=num_iter) # fc1 in [32, 512]
    opt_l = ['sgd', 'nesterov', 'adam'] # opt in {sgd, nesterov, adam}
    opt = np.random.randint(2, 3, size=num_iter)
    lr = np.random.randint(4, 5, size=num_iter) # lr in {0.001, 0.0001}
    mom = np.random.uniform(0.7, 0.91, size=num_iter) # mom in [0.7, 0.99]
    drop = np.random.uniform(0.4, 0.81, size=num_iter) # dropout in [0.2, 0.8]
    batch = 2**(np.random.randint(6, 7, size=num_iter)) # batch in [16, 64]
    
    for i in range(num_iter):
        net, losses, vals = train(S=1, P=pad[i], conv1=(1, c1[i], 5), pool=2, pool_S=2, conv2=(c1[i], c2[i], 5), 
                            fc1=fc1[i], drop=drop[i], opt=opt_l[opt[i]], lr=0.1**lr[i], mom=mom[i], batch=int(batch[i]))
        print("Trained NN{:}".format(i))
        
        # Save NN
        PATH = 'fashionmnist/fashionmnist_net{:}.pth'.format(i)
        torch.save(net.state_dict(), PATH)
        
        # Save losses
        pd.DataFrame(losses).to_csv("fashionmnist/losses{:}.csv".format(i))
        pd.DataFrame(vals).to_csv("fashionmnist/vals{:}.csv".format(i))
        # Save parameters
        with open('fashionmnist/params{:}.txt'.format(i), 'w+') as f:
            f.write("{:}\n opt: {:}\n lr: {:}\n mom: {:}\n drop: {:}\n batch: {:}".format(net.parameters, opt_l[opt[i]], 
                                                                                          0.1**lr[i], mom[i], drop[i], batch[i]))
        
        print("Saved NN{:}".format(i))

# Randomly set hyperparameters and train
# random_search(1)

# TRAINING ON VALIDATION
def train_val():
	BATCH=64
	valloader = torch.utils.data.DataLoader(val_images, batch_size=BATCH,
	                                              shuffle=True, num_workers=2)
	epsilon = 0.005
	losses = []
	vals = []
	for epoch in range(2):  # loop over the dataset multiple times 
	    running_loss = 0.0
	    for i, data in enumerate(valloader):
	        # get the inputs; data is a list of [inputs, labels]
	        inputs = data[:, :-1]
	        labels = data[:, -1]
	        # zero the parameter gradients
	        optimizer.zero_grad()

	        # forward + backward + optimize
	        if inputs.shape[0] == batch:
	            outputs = net(inputs.reshape(batch, 1, 28, 28).float())
	            outputs = outputs.reshape(batch, -1)
	            loss = criterion(outputs, labels.long())
	            loss.backward()
	            optimizer.step()

	        # print statistics
	        running_loss += loss.item()
	        if i % (10000//(10*batch)) == ((10000//(10*batch))-1):    # print every 1/10 epoch
	            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (10000//(10*batch))))
	            losses.append(running_loss/(10000//(10*batch)))
	            running_loss = 0.0
	                
	    if losses[-9] - losses[-1] < epsilon:
	        break

	print('Finished Training')
	print("Trained NN{:}".format(i))
	        
	# Save NN
	PATH = 'fashionmnist/fashionmnist_net{:}.pth'.format(100)
	torch.save(net.state_dict(), PATH)
	        
	# Save losses
	pd.DataFrame(losses).to_csv("fashionmnist/losses{:}.csv".format(100))
	pd.DataFrame(vals).to_csv("fashionmnist/vals{:}.csv".format(100))
	        
	print("Saved NN{:}".format(i))

def load():
	# Load NN
	PATH = 'fashionmnist/fashionmnist_net{:}.pth'.format(11)
	BATCH = 64

	net = Net(S=1, P=0, conv1=(1,32,5), pool=2, pool_S=2, conv2=(32,512,5), fc1=128, drop=0.3539679254488697, batch=100)
	net.load_state_dict(torch.load(PATH))

	# predictions on test set
	testloader = torch.utils.data.DataLoader(test_images, batch_size=100, shuffle=False, num_workers=2)

	predictions = []
	with torch.no_grad():
	    for i, data in enumerate(testloader):
	        outputs = net(data.reshape(100, 1, 28, 28).float())
	        predictions.extend(torch.max(outputs.data, 1)[1])
	        print(i*100)
	        

	pd.DataFrame(predictions).to_csv("predictions2.csv")


# CIFAR100
# TRAIN_SIZE, VAL_SIZE = 40000, 10000
# dataset_prefix = "cifar100" 
# num_classes = 100

# train_data = np.load("./hw5_data/{}_train.npy".format(dataset_prefix))
# test_data = np.load("./hw5_data/{}_test.npy".format(dataset_prefix))

# Note: I am unable to use sklearn at the moment due to package inconsistencies
# Split train data to train/val/test
# train_images = train_data[:TRAIN_SIZE, :]
# val_images = train_data[TRAIN_SIZE:, :]

# test_images = test_data.reshape(-1, 3, 32, 32)

def imshow(image, title=None):
    image = Image.fromarray((np.moveaxis(image, (0,1,2), (2, 0, 1))*255).astype("uint8"), 'RGB')
    fig, ax = plt.subplots(1, figsize=(2,2))
    ax.imshow(image)
    if title is not None:
        plt.title(title)
# for i in range(10):
#     imshow(train_images[i], title='Image')

# PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py


# CNN Architecture: (Conv2d->ReLU->Maxpool) x M -> (Linear->ReLU) x N -> Softmax
# Maxpool only used at most thrice to prevent erosion of signals (HxW will keep shrinking for each pool)
class CIFARNet(nn.Module):
    def __init__(self, S=1, P=0, convlist=[(10, 5), (10, 5)], numpools=2, pool=2, pool_S=2, fclist=[20, 10], drop=0.5, batch=1):
        super(Net, self).__init__()
        
        # Conv Filters list; First in_channel is 3 (RGB image)
        self.numconv = len(convlist)
        self.conv1 = nn.Conv2d(3, convlist[0][0], convlist[0][1], stride=S, padding=P)
        self.conv2 = nn.Conv2d(convlist[0][0], convlist[1][0], convlist[1][1], stride=S, padding=P)
        if self.numconv > 2:
            self.conv3 = nn.Conv2d(convlist[1][0], convlist[2][0], convlist[2][1], stride=S, padding=P)
            if self.numconv > 3:
                self.conv4 = nn.Conv2d(convlist[2][0], convlist[3][0], convlist[3][1], stride=S, padding=P)
                if self.numconv > 4:
                    self.conv5 = nn.Conv2d(convlist[3][0], convlist[4][0], convlist[4][1], stride=S, padding=P)
        
        # Maxpool
        self.numpools = numpools
        self.pool = nn.MaxPool2d(pool, stride=pool_S)
               
        # FC Layers list and batchnorms
#         print(convlist, numpools)
        self.l = int(convlist[-1][0] * (32 / (2**numpools))**2)
        self.numfc = len(fclist)+1
        self.fc1 = nn.Linear(self.l, fclist[0], bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=fclist[0])
        
        if self.numfc > 2:
            self.fc2 = nn.Linear(fclist[0], fclist[1], bias=True)
            self.bn2 = nn.BatchNorm1d(num_features=fclist[1])
            if self.numfc > 3:
                self.fc3 = nn.Linear(fclist[1], fclist[2], bias=True)
                self.bn3 = nn.BatchNorm1d(num_features=fclist[2])
                if self.numfc > 4:
                    self.fc4 = nn.Linear(fclist[2], fclist[3], bias=True)
                    self.bn4 = nn.BatchNorm1d(num_features=fclist[3])
                    
        self.fcf = nn.Linear(fclist[-1], 100, bias=True)
        
        # Softmax
        self.softmax = nn.Softmax(dim=1)
        
        # Dropout and batch size 
        self.dropout = nn.Dropout(drop)
        self.batch = batch
        print("Batch size: {:}".format(self.batch))
        

    def forward(self, x):
        # Convolution filters
        x = self.pool(F.relu(self.dropout(self.conv1(x))))
        x = F.relu(self.dropout(self.conv2(x)))
        if self.numpools > 1:
            x = self.pool(x)
        if self.numconv > 2:
            x = F.relu(self.dropout(self.conv3(x)))
            if self.numpools > 2:
                x = self.pool(x)
            if self.numconv > 3:
                x = F.relu(self.dropout(self.conv4(x)))
                if self.numpools > 3:
                    x = self.pool(x)
                if self.numconv > 4:
                    x = F.relu(self.dropout(self.conv5(x)))
                    if self.numpools > 4:
                        x = self.pool(x)
                        
        x = x.view(-1, self.l)
        
        # FC layers
        x = F.relu(self.bn1(self.fc1(x)))
        if self.numfc > 2:
            x = F.relu(self.bn2(self.fc2(x)))
            if self.numfc > 3:
                x = F.relu(self.bn3(self.fc3(x)))
                if self.numfc > 4:
                    x = F.relu(self.bn4(self.fc4(x)))
                    
        x = F.relu(self.fcf(x))
        x = self.softmax(x)
        
        return x


def cifartrain(S, P, convlist, numpools, pool, pool_S, fclist, drop, opt, lr, mom, batch):
    # NN INITIALIZATION
    net = CIFARNet(S=S, P=P, convlist=convlist, numpools=numpools, pool=pool, pool_S=pool_S, fclist=fclist, drop=drop, batch=batch)    
    
    criterion = nn.CrossEntropyLoss()
    # Using Adam Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.999))
    
    trainloader = torch.utils.data.DataLoader(train_images, batch_size=batch,
                                              shuffle=True, num_workers=2)
    epsilon = 0.0
    losses = []
    accs = []
    vals = []
    for epoch in range(100):  # loop over the dataset multiple times 
        running_loss = 0.0
        train_correct, train_total = 0, 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[:, :-1]
            labels = data[:, -1]
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if inputs.shape[0] == batch:
                outputs = net(inputs.reshape(batch, 3, 32, 32).float())
                outputs = outputs.reshape(batch, -1)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % (40000//(10*batch)) == ((40000//(10*batch))-1):    # print every 1/10 epoch
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (40000//(10*batch))))
                print('Accuracy of the network on train images: %d %%' % (
                    100 * train_correct / train_total))
                
                losses.append(running_loss/(40000//(10*batch)))
                accs.append(100 * train_correct / train_total)
                running_loss = 0.0
                
                # val accuracy
                valloader = torch.utils.data.DataLoader(val_images, batch_size=VAL_SIZE,
                                                          shuffle=True, num_workers=2)
                correct = 0
                total = 0
                with torch.no_grad():
                    
                    for data in valloader:
                        images, labels = data[:, :-1], data[:, -1]
                        outputs = net(images.reshape(VAL_SIZE, 3, 32, 32).float())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the 10000 validation images: %d %%' % (
                    100 * correct / total))
                vals.append(100 * correct / total)
                

        if (vals[-1] - vals[-9] < (epsilon-1)):
            break

    print('Finished Training')
    return net, losses, vals


# Perform random search to tune hyperparameters; more efficient than grid search since it allows for identifying important
# parameters more quickly, and avoids prioritizing less important ones (Stanford CS231)

def cifarrandom_search(num_iter=10):
    # S = 1 as good practice and simplified dimensional constraints
    # pool = 2, pool_S=2 as convention according to cs231
    kernel = (2*np.random.randint(0, 2, size=num_iter)) + 3
    pad = kernel // 2 # If conv kernel=5, pad = 2; if conv kernel=3, pad = 1; this simplifies dimensional constraints
    
    convlist = 2**np.random.randint(5, 8, size=(num_iter, 5)) # convlist.size in [3, 3], conv in [16, 128]
    numpools = np.random.randint(3, 4, size=num_iter) # numpools in [1, 3]
    
    fclist = 2**np.random.randint(6, 8, size=(num_iter, 5)) # fclist.size in [2, 3], fc in [16, 128]
    opt = 'adam'
    
    lr = np.random.randint(4, 5, size=num_iter) # lr in {0.001, 0.0001}
    mom = np.random.uniform(0.7, 0.99, size=num_iter) # mom in [0.7, 0.99]
    drop = np.random.uniform(0.4, 0.81, size=num_iter) # dropout in [0.4, 0.8]
    batch = 2**(np.random.randint(3, 7, size=num_iter)) # batch in [8, 64]
    #convlist=[(convlist[i][j], kernel[i]) for j in range(np.random.randint(3,4))]
    #fclist=fclist[i][:np.random.randint(2,3)]
    for i in range(num_iter):
        net, losses, vals = cifartrain(S=1, P=pad[i], convlist=[(16, kernel[i]), (32, kernel[i]), (64, kernel[i])],
                                  numpools=numpools[i], pool=2, pool_S=2, fclist=[128,256], 
                                  drop=drop[i], opt=opt, lr=0.1**lr[i], mom=mom[i], batch=int(batch[i]))
        
        print("Trained NN{:}".format(i+10))
        
        # Save NN
        PATH = 'cifar/cifar_net{:}.pth'.format(i+10)
        torch.save(net.state_dict(), PATH)
        
        # Save losses
        pd.DataFrame(losses).to_csv("cifar/cifarlosses{:}.csv".format(i+10))
        pd.DataFrame(vals).to_csv("cifar/cifarvals{:}.csv".format(i+10))
        # Save parameters
        with open('cifar/cifarparams{:}.txt'.format(i+10), 'w+') as f:
            f.write("{:}\n lr: {:}\n mom: {:}\n drop: {:}\n batch: {:}".format(net.parameters, 0.1**lr[i], mom[i], drop[i], batch[i]))
        
        print("Saved NN{:}".format(i+10))

# Same as fashionMNIST random_search
# cifarrandom_search(1)

def cifarpredict():
    # Load NN
    PATH = 'cifar/cifar_net{:}.pth'.format(2)
    BATCH = 100

    net = CIFARNet(S=1, P=1, convlist=[(16, 3), (32, 3), (16, 3)], numpools=3, pool=2, pool_S=2, fclist=[128, 128], 
              drop=0.5728279674128991, batch=BATCH)
    net.load_state_dict(torch.load(PATH))

    # predictions on test set
    testloader = torch.utils.data.DataLoader(test_images, batch_size=BATCH, shuffle=False, num_workers=2)

    predictions = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            outputs = net(data.reshape(BATCH, 3, 32, 32).float())
            predictions.extend(torch.max(outputs.data, 1)[1])
            print(i)
            

    pd.DataFrame(predictions).to_csv("predictions.csv")