import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if training==True:
        dataset = datasets.MNIST('./data', train=True, download=True,
                       transform=custom_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size = 50, shuffle=True)
        
    else:
        dataset = datasets.MNIST('./data', train=False, download=True,
                       transform=custom_transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size = 50, shuffle=False)
    
        
    return loader



def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    return model




def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        for idx, data in enumerate(train_loader, 0):
            images, labels = data
            opt.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            for j, i in enumerate(outputs):
                big = max(i)
                k = list(i).index(big)
                if k == labels[j]:
                    correct += 1
            
            running_loss += loss.item()
        
        
        print(f'Train Epoch: {epoch}  Accuracy: {correct}/{len(train_loader.dataset)}({round((correct*100)/len(train_loader.dataset), 2)}%)  Loss: {round((running_loss*50)/len(train_loader.dataset), 3)}')
        
        #print('Train Epoch: {:2.2%}'.format(epoch))
        
    
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cross-entropy

    RETURNS:
        None
    """
    
    model.eval()
    
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        for images, labels in test_loader:
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            for j, i in enumerate(outputs):
                big = max(i)
                k = list(i).index(big)
                if k == labels[j]:
                    correct += 1
            
            
            
        if show_loss == False:
            print(f'Accuracy: {round((correct*100)/len(test_loader.dataset), 2)}')
        else:
            
            print(f'Average loss: {round((running_loss*50)/len(test_loader.dataset), 4)}\nAccuracy: {round((correct*100)/len(test_loader.dataset), 2)}')
            
    


def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    
    image, labels = test_images.dataset[index]
    
    logits = model(image)
    
    probs = F.softmax(logits, dim=-1)
    
    vals, idxs = torch.sort(probs, descending=True)
    
    
    for idx, i in enumerate(vals[0]):
        print(f'{class_names[idxs[0][idx].item()]}: {round(i.item()*100, 2)}%')
        if idx >= 2:
            break


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
