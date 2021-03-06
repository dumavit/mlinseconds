# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
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
    #  nn.Sigmoid(),
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
            nn.Linear(sizes[idx], sizes[idx+1]) for idx in range(len(sizes)-1)
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
        self.lr = 0.00042307692307692304
        self.hidden_sizes = [400, 400, 400, 400, 400, 400, 400]

        self.momentum = 0.7923
        
        self.activations = [
            nn.Hardshrink(1.1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Hardshrink(1.1),
            nn.Sigmoid()
        ]

        # NOTE: Grids

        # self.activations_grid = [
        #     [
        #         nn.Hardshrink(k),
        #         nn.LeakyReLU(negative_slope=0.01),
        #         nn.LeakyReLU(negative_slope=0.01),
        #         nn.LeakyReLU(negative_slope=0.01),
        #         nn.LeakyReLU(negative_slope=0.01),
        #         nn.LeakyReLU(negative_slope=0.01),
        #         nn.Hardshrink(k),
        #         nn.Sigmoid()
        #     ]
        #     for k in np.linspace(0.05, 2.5, 15)
        # ]

        self.lr_grid = list(np.linspace(0.0001, 0.0007, 40))

        # self.hidden_sizes_grid = [
        #     [i, j, k, l, m, o]
        #     for i in [400]
        #     for j in [400]
        #     for k in [400]
        #     for l in [400]
        #     for m in [400]
        #     for o in [400]
        #     for p in [400]
        # ]

        # self.momentum_grid = list(np.linspace(0.7,0.9,40))

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
            if time_left < 0.1 or step > 20:
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

            if correct == total:
                # NOTE: Log stats
                stats = {
                    'step': step,
                    'loss': round(loss.item(), 5),
                    'lr': self.lr,
                    'hs': self.hidden_sizes,
                    'act': self.activations,
                    'mom': getattr(self, 'momentum', 0)
                }

                print(stats)
                if self.grid_search.enabled:
                    results.append(stats)
                break

            # calculate loss
            loss = ((output-target)**2).sum()
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
        self.number_of_cases = 20

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size).to(device)
        target = torch.FloatTensor(data_size).to(device)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(7+case, 11)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))

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
        text = '\n'.join(str(i) for i in sorted(
            results, key=operator.itemgetter('step')))
        f.write(text)
