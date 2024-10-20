import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_step(model : nn.Module,
               train_data : torch.utils.data.dataloader,
               loss_fn : nn.Module,
               optimizer : torch.optim.Optimizer,
               device : torch.device):
    model.train()
    train_loss,train_acc = 0,0
    for batch,(X,y) in enumerate(train_data):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss /= len(train_data)
    train_acc /= len(train_data)
    return train_loss,train_acc

def test_step(model:nn.Module,
               test_data : torch.utils.data.dataloader,
               loss_fn : nn.Module,
               device:torch.device):
    model.eval()
    test_loss,test_acc = 0,0

    with torch.inference_mode():

        for batch,(X,y) in enumerate(test_data):
            X,y = X.to(device),y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred,y)
            test_loss += loss.item()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (y_pred_class == y).sum().item()/len(y_pred)

    test_loss /= len(test_data)
    test_acc /= len(test_data)
    return test_loss,test_acc

def train(model:nn.Module,
              train_data : torch.utils.data.dataloader,
              test_data : torch.utils.data.dataloader,
              loss_fn : nn.Module,
              optimizer : torch.optim.Optimizer,
              epochs : int,
              device:torch.device):
    result = {'epoch':[],
                'train_loss':[],
                'train_acc':[],
                'test_loss':[],
                'test_acc':[]}

    for epoch in tqdm(range(epochs)):
        train_loss,train_acc = train_step(model = model,
                                train_data = train_data,
                                loss_fn = loss_fn,
                                optimizer = optimizer,
                                          device=device)
        test_loss,test_acc = test_step(model = model,
                                test_data = test_data,
                                loss_fn = loss_fn,
                                       device=device)
        print(
              f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}"
          )

        result['epoch'].append(epoch+1)
        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc)
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc)

    return result

