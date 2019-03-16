# There are 2 functions defined from input. One easy one and one hard one.
# On training data easy and hard functions produce same result and on
# test data you need to predict hard function.
# Easy function - depends on fewer inputs.
# Hard function - depends on more inputs.
# Easy and hard function depends on different inputs.
# Functions is a random functions of n-inputs, it's guarantee that
# functions depends on n inputs.
# For example:
# Inputs:
# x0, x1, x2, x3, x4, x5, x6, x7
# Easy function:
# Hard function:
# x2^x3^x4^x5^x6^x7
# x0^x1
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..utils import solutionmanager as sm
from ..utils.gridsearch import GridSearch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 1)

        self.batch_norm = nn.BatchNorm1d(8, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


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
        self.lr = 0.00723571

        self.hidden_sizes = [24, 24, 24]
        self.momentum = 0.7

        self.activations = [
            nn.ReLU6(),
            nn.ReLU6(),
            nn.ReLU6(),
            nn.Sigmoid()
        ]

        # NOTE: Grids
        self.lr_grid = list(np.linspace(0.0001, 0.1, 15)) + list(np.linspace(0.2, 1.5, 10))

        self.hidden_sizes_grid = [
            [i, i, i]
            for i in [24]
        ]

        # self.momentum_grid = [0, 0.11, 0.357, 0.72]

        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)

    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, target, context):
        step = 0
        # Put model in train mode
        model.to(device)
        model.train()

        simple_model = SimpleModel()
        simple_model.to(device)
        simple_optimizer = optim.RMSprop(simple_model.parameters(), lr=0.01, momentum=0.7)

        simple_model_results = [0] * 8

        # Train simple model to learn easy function inputs
        for i in range(10):
            for col_index in range(8):
                data = train_data.clone()
                # Inverse one input (0->1, 1->0)
                data[:, col_index] = torch.abs(data[:, col_index] - 1)

                simple_optimizer.zero_grad()
                output = simple_model(data)
                predict = output.round()
                correct = predict.eq(target.view_as(predict)).long().sum().item()

                loss = ((output - target) ** 2).sum()
                loss.backward()
                simple_optimizer.step()
                simple_model_results[col_index] = correct

        # When easy input is inverted - model gets the worst results
        easy_input_1, easy_input_2 = np.argpartition(simple_model_results, 2)[:2]
        hard_data = train_data.clone()

        # Delete easy inputs to learn hard function
        hard_data[:, easy_input_1] = 0
        hard_data[:, easy_input_2] = 0

        optimizer = optim.RMSprop(model.parameters(), lr=self.lr, momentum=self.momentum)
        total = target.view(-1).size(0)

        while True:
            time_left = context.get_timer().get_time_left()
            if time_left < 0.1:
                break

            optimizer.zero_grad()
            output = model(hard_data)
            predict = output.round()
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            loss = ((output - target) ** 2).sum()

            if correct == total:
                break

            loss.backward()
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
        self.test_limit = 0.75


class DataProvider:
    def __init__(self):
        self.number_of_cases = 20

    def full_func(self, input_size):
        while True:
            table = torch.ByteTensor(1<<input_size).random_(0, 2)
            vals = torch.ByteTensor(input_size, 2).zero_()
            depend_count = 0
            for i in range(input_size):
                for ind in range(1<<input_size):
                    if table[ind].item() != table[ind^(1<<i)].item():
                        depend_count += 1
                        break
            if depend_count == input_size:
                return table

    def tensor_to_int(self, tensor):
        tensor = tensor.view(-1)
        res = 0
        for x in tensor:
            res = (res<<1)+x.item()
        return res

    def int_to_tensor(self, ind, tensor):
        for i in range(tensor.size(0)):
            tensor[i] = (ind >> i)&1

    def create_data(self, seed, easy_table, hard_table, easy_input_size, hard_input_size, easy_correct):
        input_size = easy_input_size + hard_input_size
        data_size = 1 << input_size
        data = torch.ByteTensor(data_size, input_size).to(device)
        target = torch.ByteTensor(data_size, 1).to(device)
        count = 0
        for ind in range(data_size):
            self.int_to_tensor(ind, data[count])
            easy_ind = ind & ((1 << easy_input_size)-1)
            hard_ind = ind >> easy_input_size
            easy_value = easy_table[easy_ind].item()
            hard_value = hard_table[hard_ind].item()
            target[count, 0] = hard_value
            if not easy_correct or easy_value == hard_value:
                count += 1
        data = data[:count,:]
        target = target[:count,:]
        perm = torch.randperm(count)
        data = data[perm]
        target = target[perm]
        return (data.float(), target.float())

    def create_case_data(self, case):
        easy_input_size = 2
        hard_input_size = 6

        random.seed(case)
        torch.manual_seed(case)
        easy_table = self.full_func(easy_input_size)
        hard_table = self.full_func(hard_input_size)
        train_data, train_target = self.create_data(case, easy_table, hard_table, easy_input_size, hard_input_size, True)
        test_data, test_target = self.create_data(case, easy_table, hard_table, easy_input_size, hard_input_size, False)
        perm = torch.randperm(train_data.size(1))
        train_data = train_data[:,perm]
        test_data = test_data[:,perm]
        return sm.CaseData(case, Limits(), (train_data, train_target), (test_data, test_target)).set_description("Easy {} inputs and hard {} inputs".format(easy_input_size, hard_input_size))

class Config:
    def __init__(self):
        self.max_samples = 10000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
