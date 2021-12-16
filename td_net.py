import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# # class TD_SGD(torch.optim.SGD):
# #     def __init__(self, params, lr=0.1, momentum=0.7):
# #         super(TD_SGD, self).__init__(params, lr, momentum=momentum)

# #     def step(self, losses, lrs, closure=None):
# #         loss = None
# #         if closure is not None:
# #             with torch.enable_grad():
# #                 loss = closure()

# #         for group in self.param_groups:
# #             params_with_grad = []
# #             d_p_list = []
# #             momentum_buffer_list = []
# #             weight_decay = group['weight_decay']
# #             momentum = group['momentum']
# #             dampening = group['dampening']
# #             nesterov = group['nesterov']
# #             maximize = group['maximize']
# #             lr = group['lr']

# #             for p in group['params']:
# #                 if p.grad is not None:
# #                     params_with_grad.append(p)
# #                     d_p_list.append(p.grad)

# #                     state = self.state[p]
# #                     if 'momentum_buffer' not in state:
# #                         momentum_buffer_list.append(None)
# #                     else:
# #                         momentum_buffer_list.append(state['momentum_buffer'])

# #             F.sgd(params_with_grad,
# #                   d_p_list,
# #                   momentum_buffer_list,
# #                   weight_decay=weight_decay,
# #                   momentum=momentum,
# #                   lr=lr,
# #                   dampening=dampening,
# #                   nesterov=nesterov,
# #                   maximize=maximize,)

# #             # update momentum_buffers in state
# #             for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
# #                 state = self.state[p]
# #                 state['momentum_buffer'] = momentum_buffer

# #         return loss


# def Loss_Y0(pred, target):
#     print(pred.requires_grad)
#     mask = torch.Tensor([1, 0, 0, 0])
#     return torch.sub(torch.matmul(pred, mask), torch.matmul(target, mask))


# def Loss_Y1(pred, target):
#     return torch.sub(pred[1], target[1])


# def Loss_Y2(pred, target):
#     return torch.sub(pred[2], target[2])


# def Loss_Y3(pred, target):
#     return torch.sub(pred[3], target[3])


def LinearLoss(pred, target):
    return torch.sub(pred, target)


class TD_Net(nn.Module):

    def __init__(self, input_dim=198, hidden_dim=80, output_dim=4, alpha=0.1,
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
        # self.w1.weight = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        # self.w1.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.w2 = nn.Linear(hidden_dim, output_dim)
        # self.w2.weight = nn.Parameter(torch.zeros(output_dim, hidden_dim))
        # self.w2.bias = nn.Parameter(torch.zeros(output_dim))
        self.to(device)
        self.device = device

    def forward(self, x):
        x = torch.sigmoid(self.w1(x))
        return torch.softmax(self.w2(x), dim=-1)
        # return torch.sigmoid(self.w2(x))

    def f0(self, x):
        x = torch.sigmoid(self.w1(x))
        return self.w2(x)[0]

    def f0(self, x):
        x = torch.sigmoid(self.w1(x))
        return self.w2(x)[1]

    def f0(self, x):
        x = torch.sigmoid(self.w1(x))
        return self.w2(x)[2]

    def f0(self, x):
        x = torch.sigmoid(self.w1(x))
        return self.w2(x)[3]


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
        # self.loss = nn.SmoothL1Loss()
        # self.loss = nn.CrossEntropyLoss()
        self.loss = LinearLoss
        self.device = device
        self.lambda_param = lambda_param
        self.output_dim = output_dim

    def train_single_example(self, start_state, next_state, optimizer, grads):
        '''
        This function should be used once a move has been taken by the player in
        the game
        '''
        # print(f'start state grad: {start_state.grad}')
        start_state.retain_grad()
        # print(start_state.requires_grad)
        y_0 = self.net(start_state)
        # print(f'start state grad: {start_state.grad}')
        with torch.no_grad():
            y_1 = self.net(next_state)
        # print(start_state)
        # print(next_state)
        # print(y_0)
        # print(y_1)
        total_loss = torch.zeros(4)
        for i in range(len(grads)):
            optimizer.zero_grad()
            total_loss[i] = self.loss(y_0[i], y_1[i])
            # print(total_loss[i])
            total_loss[i].backward(retain_graph=True)
            idx = 0
            for p in self.net.parameters():
                # print(p.grad)
                grads[i][idx] *= self.lambda_param
                grads[i][idx] += p.grad
                idx += 1
        idx = 0
        for p in self.net.parameters():
            p.grad = sum([total_loss[i] * grads[i][idx]
                          for i in range(self.output_dim)])
            # print(p.grad)
            idx += 1
        # print(list(self.net.parameters())[3].grad)
        optimizer.step()

        return torch.sum(torch.abs(total_loss)), grads

    def train_final_example(self, start_state, final_state_value, optimizer, grads):
        '''
        This function should be used once a move has been taken by the player in
        the game
        '''
        # print(f'start state grad: {start_state.grad}')
        start_state.retain_grad()
        # print(start_state.requires_grad)
        y_0 = self.net(start_state)
        # print(f'start state grad: {start_state.grad}')
        y_1 = final_state_value.to(self.device)
        # print(start_state)
        # print(next_state)
        print(y_0)
        print(y_1)
        total_loss = torch.zeros(4)
        for i in range(len(grads)):
            optimizer.zero_grad()
            total_loss[i] = self.loss(y_0[i], y_1[i])
            # print(total_loss[i])
            total_loss[i].backward(retain_graph=True)
            idx = 0
            for p in self.net.parameters():
                # print(p.grad)
                grads[i][idx] *= self.lambda_param
                grads[i][idx] += p.grad
                idx += 1
        idx = 0
        for p in self.net.parameters():
            p.grad = sum([total_loss[i] * grads[i][idx]
                          for i in range(self.output_dim)])
            # print(p.grad)
            idx += 1
        # print(list(self.net.parameters())[3].grad)
        optimizer.step()

        return torch.sum(torch.abs(total_loss))

    def get_predicted_move_vals(self, state_data_loader):
        batch_size = state_data_loader.batch_size
        values = torch.zeros(len(state_data_loader.dataset))
        for i, state in enumerate(state_data_loader):
            state.to(self.device)
            pred = self.net(state)
            val = torch.tensor(
                [2, 1, -1, -2], dtype=torch.float) @ pred.type(torch.FloatTensor).T
            values[i * batch_size: i * batch_size + len(state)] = val
        return values

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# # net = TD_Net(device=device)
# wrapper = TD_Net_Wrapper(device=device)
# params = wrapper.net.parameters()
# # optimizers = [torch.optim.SGD(
# #     wrapper.net.parameters(), lr=0.1, momentum=0.7) for i in range(4)]
# # for o in optimizers:
# #     o.zero_grad()
# optimizer = torch.optim.SGD(wrapper.net.parameters(), lr=0.1)
# grads = [[torch.zeros((80, 198)), torch.zeros(80), torch.zeros((4, 80)), torch.zeros(4)]
#          for i in range(4)]

# # print(params)
# for p in params:
#     print(p.size())
# x_1 = torch.ones(198, requires_grad=True)
# # # not sure if this is necessary
# x_1.to(device)
# x_2 = torch.zeros(198, requires_grad=True)
# # # not sure this is necessary
# x_2.to(device)

# wrapper.train_single_example(x_1, x_2, optimizer, grads)
