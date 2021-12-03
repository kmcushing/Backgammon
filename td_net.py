import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TD_Net(nn.Module):

    def __init__(self, input_dim=52, hidden_dim=80, output_dim=4, alpha=0.01,
                 lambda_param=0, device='cpu'):
        '''
        Defines a neural network with a single hidden layer to train according 
        to the temporal difference learning algorithm TD-Lambda

        Params:

        input_dim - size of the input layer
        hidden_dim - size of hidden layer
        output_dim - size of output layer
        alpha - learning rate (alpha in TD-Gammon paper)
        lambda_param - momentum of SGD optimizer in model (lambda in TD-Gammon)

        '''
        super(TD_Net, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim)
        self.to(device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha,
                                         momentum=lambda_param)
        # loss function in TD-Gammon paper is MAE Loss
        self.loss = nn.L1Loss()

    def forward(self, x):
        x = torch.sigmoid(self.w1(x))
        return F.softmax(self.w2(x), dim=-1)

    def train_single_example(self, start_state, next_state):
        '''
        This function should be used once a move has been taken by the player in
        the game
        '''
        y_0 = self.forward(start_state)
        with torch.no_grad():
            y_1 = self.forward(next_state)

        loss = self.loss(y_0, y_1)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train_final_example(self, start_state, final_state_value):
        '''
        This function should be used once a move has been taken by the player in
        the game
        '''
        y_0 = self.forward(start_state)

        loss = self.loss(y_0, final_state_value)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_predicted_move_vals(self, state_data_loader):
        batch_size = state_data_loader.batch_size
        values = torch.zeros(len(state_data_loader.dataset))
        for i, state in enumerate(state_data_loader):
            pred = self.forward(state)
            val = torch.tensor(
                [2, 1, -1, -2], dtype=torch.float) @ pred.type(torch.FloatTensor).T
            values[i * batch_size: i * batch_size + len(state)] = val
        return values

    def zero_grad(self):
        '''
        Resets gradient of optimizer. This should be used when beginning a new
        game to clear the gradients accumulated from the previous game.
        '''
        self.optimizer.zero_grad()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = TD_Net(device=device)

x_1 = torch.zeros(52, requires_grad=True)
# not sure if this is necessary
x_1.to(device)
x_2 = torch.ones(52, requires_grad=True)
# not sure this is necessary
x_2.to(device)

net.train_single_example(x_1, x_2)
