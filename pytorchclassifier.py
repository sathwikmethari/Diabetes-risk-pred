import torch
import torch.nn as nn

def encoding(data_copy,cat_columns):
    for column in cat_columns:
        if column=='class':
            data_copy[column]=data_copy[column].map({'Positive':1,'Negative':0})
            continue
        if column=='Gender':
            data_copy[column]=data_copy[column].map({'Male':0,'Female':1})
            continue
        data_copy[column]=data_copy[column].map({'No':0,'Yes':1})
        
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

class ClassificationModel(nn.Module):
    def __init__(self,input_shape,hidden_shape):
        super().__init__()
        self.stack1=nn.Sequential(
            nn.Linear(in_features=input_shape,out_features=hidden_shape),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape,out_features=hidden_shape*4),
        )
        self.stack2=nn.Sequential(
            nn.Linear(in_features=hidden_shape*4,out_features=hidden_shape),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape,out_features=hidden_shape*2),
        )
        self.stack3=nn.Sequential(
            nn.Linear(in_features=hidden_shape*2,out_features=hidden_shape),
            nn.ReLU(),
            nn.Linear(hidden_shape,1),
        )
    def forward(self,x):
        x=self.stack1(x)
        x=self.stack2(x)
        x=self.stack3(x)
        return x