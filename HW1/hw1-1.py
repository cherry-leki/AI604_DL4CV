import torch
import torch.nn as nn #
import torch.nn.functional as F # various activation functions for model
import torchvision # You can load various Pretrained Model from this package
import torchvision.datasets as vision_dsets
import torchvision.transforms as T # Transformation functions to manipulate images
import torch.optim as optim # various optimization functions for model
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils import data
from sklearn.metrics import confusion_matrix

def MNIST_DATA(root='./data',train =True,transforms=None ,download =True,batch_size = 32,num_worker = 1):
    print ("[+] Get the MNIST DATA")
    """
    We will use Mnist data for our tutorial 
    """
    mnist_train = vision_dsets.MNIST(root = root,  #root is the place to store your data.
                                    train = True,
                                    transform = T.ToTensor(), # convert data to tensor
                                    download = True)  # whether to download the data
    mnist_test = vision_dsets.MNIST(root = root,
                                    train = False,
                                    transform = T.ToTensor(),
                                    download = True)
    """
    Data Loader is a iterator that fetches the data with the number of desired batch size. 
    * Practical Guide : What is the optimal batch size? 
      - Usually.., higher the batter. 
      - We recommend to use it as a multiple of 2 to efficiently utilize the gpu memory. (related to bit size)
    """
    trainDataLoader = data.DataLoader(dataset = mnist_train,  # information about your data type
                                      batch_size = batch_size, # batch size
                                      shuffle =True, # Whether to shuffle your data for every epoch. (Very important for training performance)
                                      num_workers = 1) # number of workers to load your data. (usually number of cpu cores)

    testDataLoader = data.DataLoader(dataset = mnist_test,
                                    batch_size = batch_size,
                                    shuffle = False, # we don't actually need to shuffle data for test
                                    num_workers = 1) #
    print ("[+] Finished loading data & Preprocessing")
    return mnist_train,mnist_test,trainDataLoader,testDataLoader

class Trainer():
    def __init__(self, trainloader, testloader, net, optimizer, criterion):
        """
        trainloader: train data's loader
        testloader: test data's loader
        net: model to train
        optimizer: optimizer to update your model
        criterion: loss function
        """
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, epoch=1):
        """
        epoch: number of times each training sample is used
        """
        self.net.train()
        for e in range(epoch):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data  # Return type for data in dataloader is tuple of (input_data, labels)
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)  # get output after passing through the network
                loss = self.criterion(outputs, labels)  # compute model's score using the loss function
                loss.backward()  # perform back-propagation from the loss
                self.optimizer.step()  # perform gradient descent with given optimizer

                # print statistics
                running_loss += loss.item()
                if (i + 1) % 500 == 0:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (e + 1, i + 1, running_loss / 500))
                    running_loss = 0.0

        print('Finished Training')

    def test(self):
        self.net.eval() 
        test_loss = 0
        correct = 0
        
        # Data for confusion matrix
        conf_true = torch.zeros(0, dtype=torch.long, device='cpu')
        conf_pred = torch.zeros(0, dtype=torch.long, device='cpu')

        for inputs, labels in self.testloader:
            inputs = inputs.cuda()
            labels = labels.cuda() 
            output = self.net(inputs) 
            pred = output.max(1, keepdim=True)[1] # get the index of the max 
            correct += pred.eq(labels.view_as(pred)).sum().item()

            test_loss /= len(self.testloader.dataset)
            
            # Append batch prediction results for confusion matrix
            conf_true = torch.cat([conf_true, labels.view(-1).cpu()])
            conf_pred = torch.cat([conf_pred, pred.view(-1).cpu()])
                
        print('\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.
                format(correct, len(self.testloader.dataset),
                100.* correct / len(self.testloader.dataset)))

        # Print confusion matrix
        conf_mat = confusion_matrix(conf_true.numpy(), conf_pred.numpy())
        print('\nConfusion matrix\n')
        print(conf_mat)

        # per-class accuracy
        conf_class_acc = 100 * conf_mat.diagonal()/conf_mat.sum(1)
        print('\nAccuracy per class\n')
        for i in range(10):
            print(i, 'th acc: ', conf_class_acc[i])
        
        print('\nClassification report\n')
        print(classification_report(conf_true.numpy(), conf_pred.numpy(), target_names=[str(i) for i in range(10)]))

    def compute_conf(self):
        self.net.eval()
        
        # Data for confusion matrix
        conf_true = torch.zeros(0, dtype=torch.long, device='cpu')
        conf_pred = torch.zeros(0, dtype=torch.long, device='cpu')

        for inputs, labels in self.testloader:
            inputs = inputs.cuda()
            labels = labels.cuda() 
            output = self.net(inputs) 
            pred = output.max(1, keepdim=True)[1] # get the index of the max 
            
            # Append batch prediction results for confusion matrix
            conf_true = torch.cat([conf_true, labels.view(-1).cpu()])
            conf_pred = torch.cat([conf_pred, pred.view(-1).cpu()])

        # Print confusion matrix
        label_name = [str(i) for i in range(10)]
        conf_mat = confusion_matrix(conf_true.numpy(), conf_pred.numpy(), labels=[i for i in range(10)])
        print('\nConfusion matrix\n')
        print(conf_mat)

        # per-class accuracy
        conf_class_acc = 100 * conf_mat.diagonal()/conf_mat.sum(1)
        print('\nAccuracy per class\n')
        for i in range(10):
            print(i, 'th acc: ', conf_class_acc[i])


        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf_mat)
        fig.colorbar(cax)
        ax.set_xticks([i for i in range(10)])
        ax.set_yticks([i for i in range(10)])
        ax.set_xticklabels(label_name)
        ax.set_yticklabels(label_name)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        


class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        # self.conv0 = nn.Conv2d(1, 8, 5, 1)
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1)
        self.conv0_bn = nn.BatchNorm2d(8)
        self.pool0 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        # self.conv1 = nn.Conv2d(8, 16, 5, 1)
        self.conv1 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        # self.fc = nn.Linear(16*4*4, 10)
        self.fc = nn.Linear(16*7*7, 256)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv0_bn(x)
        x = F.relu(x)
        x = self.pool0(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc1(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    trainDset, testDset, trainDataLoader, testDataLoader = MNIST_DATA(batch_size=32)  # Data Loader
    mnist_net = MNIST_Net().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)

    trainer = Trainer(trainloader = trainDataLoader,
                  testloader = testDataLoader,
                  net = mnist_net,
                  criterion = criterion,
                  optimizer = optimizer)

    trainer.train(epoch = 4)
    trainer.test()

    trainer.compute_conf()
    print(count_parameters(mnist_net))