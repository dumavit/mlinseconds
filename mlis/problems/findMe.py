# There are random function from 8 inputs and X random inputs added.
# We split data in 2 parts, on first part you will train and on second
# part we will test
import time
import random
import operator
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ..utils import solutionmanager as sm
from ..utils.gridsearch import GridSearch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activations = [
    # nn.Sigmoid(),
    #  nn.LogSigmoid(),
    nn.ReLU6(),
    nn.LeakyReLU(negative_slope=0.01),
    # nn.ELU(),
    # nn.SELU(),
    # nn.Hardtanh(),
    #   nn.Hardshrink(),
    nn.Hardshrink(1),
]


class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, params):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.params = params

        sizes = [input_size] + params.hidden_sizes + [output_size]

        self.layers = nn.ModuleList(
            nn.Linear(sizes[idx], sizes[idx + 1]) for idx in range(len(sizes) - 1)
        ).to(device)

        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(size, affine=False, track_running_stats=False) for size in sizes[1:]
        )

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)

            if idx != len(self.layers) - 1:
                x = self.batch_norms[idx](x)
            x = self.params.activations[idx](x)
        return x


class Solution():
    def __init__(self):
        # NOTE: Network params
        self.lr = 2.759344827586207
        self.hidden_sizes = [16, 16, 10, 10, 10]
        self.momentum = 0.9
        self.activations = [nn.LeakyReLU(), nn.ReLU6(), nn.ReLU6(), nn.ReLU6(), nn.ReLU6(), nn.Sigmoid()]

        # NOTE: Grids
        self.activations_grid = [
            [i, j, nn.ReLU6(), nn.ReLU6(), nn.ReLU6(), nn.Sigmoid()]
            for i in activations
            for j in activations
        ]

        self.lr_grid = list(np.linspace(0.001, 10, 30))

        self.hidden_sizes_grid = [
            [i, j, k, l, m]
            for i in [12, 16]
            for j in [12, 16]
            for k in [8, 10, 12]
            for l in [8, 10]
            for m in [8, 10]
        ]

        self.momentum_grid = list(np.linspace(0.5, 1, 10))

        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.to(device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)

        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1 or step > 88:
                break
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = output.round()
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = target.view(-1).size(0)
            # calculate loss
            bce_loss = nn.BCELoss()
            loss = bce_loss(output, target)

            if correct == total and loss.item() < 0.00025:
                # NOTE: Log stats
                stats = {
                    'step': step,
                    'loss': round(loss.item(), 5),
                    'lr': self.lr,
                    'hs': self.hidden_sizes,
                    'act': self.activations,
                    'mom': getattr(self, 'momentum', 0)
                }
                if self.grid_search.enabled:
                    results.append(stats)
                break

            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step


###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 1000000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, data_size, input_size, random_input_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_input = torch.ByteTensor(function_size, input_size)
        for i in range(function_input.size(0)):
            fun_ind = i
            for j in range(function_input.size(1)):
                input_bit = fun_ind&1
                fun_ind = fun_ind >> 1
                function_input[i][j] = input_bit
        function_output = torch.ByteTensor(function_size).random_(0, 2)

        if data_size % function_size != 0:
            raise "Data gen error"

        data_input = torch.ByteTensor(data_size, input_size).view(-1, function_size, input_size)
        target = torch.ByteTensor(data_size).view(-1, function_size)
        for i in range(data_input.size(0)):
            data_input[i] = function_input
            target[i] = function_output
        data_input = data_input.view(data_size, input_size)
        target = target.view(data_size)
        if random_input_size > 0:
            data_random = torch.ByteTensor(data_size, random_input_size).random_(0, 2)
            data = torch.cat([data_input, data_random], dim=1)
        else:
            data = data_input
        perm = torch.randperm(data.size(1))
        data = data[:,perm]
        perm = torch.randperm(data.size(0))
        data = data[perm].to(device)
        target = target[perm].to(device)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        data_size = 256*32
        input_size = 8
        random_input_size = min(32, (case-1)*4)

        data, target = self.create_data(2*data_size, input_size, random_input_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs and {} random inputs".format(input_size, random_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()


# If you want to run specific case, put number here
results = []
sm.SolutionManager(Config()).run(case_number=-1)

if results:
    with open('results.txt', 'w') as f:
        text = ',\n'.join(str(i) for i in sorted(
            results, key=operator.itemgetter('step')))
        f.write(text)
