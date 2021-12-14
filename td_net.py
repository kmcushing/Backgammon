import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TD_Net(nn.Module):

    def __init__(self, input_dim=52, hidden_dim=80, output_dim=4, alpha=0.1,
                 lambda_param=0.7, device='cpu'):
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
        self.w1.weight = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.w1.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Linear(hidden_dim, output_dim)
        self.w2.weight = nn.Parameter(torch.zeros(output_dim, hidden_dim))
        self.w2.bias = nn.Parameter(torch.zeros(output_dim))
        self.to(device)
        self.device = device

    def forward(self, x):
        x = torch.sigmoid(self.w1(x))
        return F.softmax(self.w2(x), dim=-1)
        # return F.normalize(self.w2(x), dim=-1)
        # return torch.sigmoid(self.w2(x))
        # return self.w2(x)


class TD_Net_Wrapper():
    def __init__(self, input_dim=198, hidden_dim=80, output_dim=4, alpha=0.1,
                 lambda_param=0.7, device='cpu'):

        self.net = TD_Net(input_dim=input_dim, hidden_dim=hidden_dim,
                          output_dim=output_dim, alpha=alpha,
                          lambda_param=lambda_param, device=device)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=alpha,
        #                                  momentum=lambda_param)
        # print(self.net.parameters())
        # print([len(l) for l in self.net.parameters()])
        # print([l.size() for l in self.net.parameters()])
        # loss function in TD-Gammon paper is MAE Loss
        # self.loss = nn.L1Loss()
        self.loss = nn.SmoothL1Loss()
        # self.loss = nn.CrossEntropyLoss()
        self.device = device

    def train_single_example(self, start_state, next_state, optimizer):
        '''
        This function should be used once a move has been taken by the player in
        the game
        '''
        # print(f'start state grad: {start_state.grad}')
        start_state.retain_grad()
        y_0 = self.net(start_state)
        # print(f'start state grad: {start_state.grad}')
        with torch.no_grad():
            y_1 = self.net(next_state)
        # print(start_state)
        # print(next_state)
        # print(y_0)
        # print(y_1)
        a = list(self.net.parameters())[0].clone()
        # print([p.shape for p in a])
        loss = self.loss(y_0, y_1)
        # print(loss)
        # print(self.net.grad_fn)
        # print(loss.grad_fn)
        # print(f'start state grad: {start_state.grad}')
        loss.backward()
        # print(loss.grad)
        # print(f'start state grad: {start_state.grad}')
        optimizer.step()
        # b = list(self.net.parameters())[3].clone()
        # print(b)
        # print(list(self.net.parameters())[3].grad)
        # print(f'start state grad: {start_state.grad}')
        # print(torch.equal(a, b))
        # print(self.optimizer)
        return loss.item()

    def train_final_example(self, start_state, final_state_value, optimizer):
        '''
        This function should be used once a move has been taken by the player in
        the game
        '''
        start_state.retain_grad()
        y_0 = self.net(start_state.to(self.device))

        # print(start_state)
        # print(y_0)
        # print(final_state_value)
        a = list(self.net.parameters())[0].clone()
        loss = self.loss(y_0, final_state_value.to(self.device))
        # print(loss)
        loss.backward()
        optimizer.step()
        b = list(self.net.parameters())[3].clone()
        # print(b)
        # print(list(self.net.parameters())[3].grad)
        # print(start_state.grad)
        # print(torch.equal(a, b))
        return loss.item()

    def get_predicted_move_vals(self, state_data_loader):
        batch_size = state_data_loader.batch_size
        values = torch.zeros(len(state_data_loader.dataset))
        for i, state in enumerate(state_data_loader):
            pred = self.net(state)
            val = torch.tensor(
                [2, 1, -1, -2], dtype=torch.float) @ pred.type(torch.FloatTensor).T
            values[i * batch_size: i * batch_size + len(state)] = val
        return values

    # def zero_grad(self):
    #     '''
    #     Resets gradient of optimizer. This should be used when beginning a new
    #     game to clear the gradients accumulated from the previous game.
    #     '''
    #     self.optimizer.zero_grad()


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# net = TD_Net(device=device)

# x_1 = torch.zeros(52, requires_grad=True)
# # not sure if this is necessary
# x_1.to(device)
# x_2 = torch.ones(52, requires_grad=True)
# # not sure this is necessary
# x_2.to(device)

# net.train_single_example(x_1, x_2)
