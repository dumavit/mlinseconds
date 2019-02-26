# There are random function from 8 inputs.
# There are random input vector of size 8 * number of voters.
# We calculate function number of voters times and sum result.
# We return 1 if sum > voters/2, 0 otherwise
# We split data in 2 parts, on first part you will train and on second
# part we will test
import operator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..utils import solutionmanager as sm
from ..utils.gridsearch import GridSearch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, params):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.params = params
        self.voters_count = input_size // 8

        sizes = [8] + params.hidden_sizes + [output_size]

        self.layers = nn.ModuleList(
            nn.Linear(sizes[idx], sizes[idx + 1]) for idx in range(len(sizes) - 1)
        ).to(device)

        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(size, affine=False, track_running_stats=False) for size in sizes[1:]
        )

        sizes_2 = [int(input_size / 8)] + params.hidden_sizes_2 + [1]

        self.layers_2 = nn.ModuleList(
            nn.Linear(sizes_2[idx], sizes_2[idx + 1]) for idx in range(len(sizes_2) - 1)
        ).to(device)

        self.batch_norms_2 = nn.ModuleList(
            nn.BatchNorm1d(size, affine=False, track_running_stats=False) for size in sizes_2[1:]
        )

    def forward_voter(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)

            if idx != len(self.layers) - 1:
                x = self.batch_norms[idx](x)
            x = self.params.activations[idx](x)
        return x

    def forward(self, x):
        x = x.view(x.size(0) * self.voters_count, -1)
        x = self.forward_voter(x)
        x = x.view(x.size(0) // self.voters_count, -1)

        for idx, layer in enumerate(self.layers_2):
            x = layer(x)

            if idx != len(self.layers_2) - 1:
                x = self.batch_norms_2[idx](x)
            x = self.params.activations_2[idx](x)
        return x


class Solution():
    def __init__(self):
        # NOTE: Network params
        self.lr = 0.99
        self.lr_8 = 1.1353535353535353

        self.hidden_sizes = [16, 16]
        self.hidden_sizes_2 = [40]

        self.momentum = 0.8673469387755102

        self.activations = [
            nn.ReLU6(),
            nn.ReLU6(),
            nn.Sigmoid()
        ]
        self.activations_2 = [
            nn.ReLU6(),
            nn.Sigmoid()
        ]

        # NOTE: Grids
        self.lr_grid = list(np.linspace(0.8, 1.2, 100))
        # self.activations_2_grid = [
        #     [i, nn.Sigmoid()]
        #     for i in activations
        # ]

        # self.momentum_grid = list(np.linspace(0.5, 1, 50))

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

        lr = self.lr_8 if model.voters_count == 8 else self.lr
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.momentum)

        test_data = context.case_data.test_data[0]
        test_target = context.case_data.test_data[1]

        batches = 8
        batch_size = train_data.shape[0] // batches

        while True:
            ind = step % batches
            start_ind = batch_size * ind
            end_ind = batch_size * (ind + 1)
            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]

            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                print(f'Failed step: {step}, loss: {loss.item()} ')
                break

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

            if correct == total:
                # NOTE: Log stats
                # print('trained', end=' ')
                # stats = {
                #     'step': step,
                #     'loss': round(loss.item(), 5),
                #     'lr': self.lr,
                #     'hs': self.hidden_sizes,
                #     'hs2': self.hidden_sizes_2,
                #     'act': self.activations,
                #     'act2': self.activations_2,
                #     'mom': getattr(self, 'momentum', 0)
                # }

                model.eval()
                test_output = model(test_data)
                test_predict = test_output.round()
                test_correct = test_predict.eq(test_target.view_as(test_predict)).long().sum().item()
                test_total = test_target.view(-1).size(0)
                if test_correct == test_total:
                    print('\n-------------------------DONE---------------------')
                    # print(stats)
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

    def get_index(self, tensor_index):
        index = 0
        for i in range(tensor_index.size(0)):
            index = 2*index + tensor_index[i].item()
        return index

    def calc_value(self, input_data, function_table, input_size, input_count_size):
        count = 0
        for i in range(input_count_size):
            count += function_table[self.get_index(input_data[i*input_size: (i+1)*input_size])].item()
        if count > input_count_size/2:
            return 1
        else:
            return 0

    def create_data(self, data_size, input_size, input_count_size, seed):
        torch.manual_seed(seed)
        function_size = 1 << input_size
        function_table = torch.ByteTensor(function_size).random_(0, 2)
        total_input_size = input_size*input_count_size
        data = torch.ByteTensor(data_size, total_input_size).random_(0, 2).to(device)
        target = torch.ByteTensor(data_size).to(device)
        for i in range(data_size):
            target[i] = self.calc_value(data[i], function_table, input_size, input_count_size)
        return (data.float(), target.view(-1, 1).float())

    def create_case_data(self, case):
        input_size = 8
        data_size = (1<<input_size)*32
        input_count_size = case

        data, target = self.create_data(2*data_size, input_size, input_count_size, case)
        return sm.CaseData(case, Limits(), (data[:data_size], target[:data_size]), (data[data_size:], target[data_size:])).set_description("{} inputs per voter and {} voters".format(input_size, input_count_size))

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
