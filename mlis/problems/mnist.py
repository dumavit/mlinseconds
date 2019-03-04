# The MNIST database of handwritten digits. http://yann.lecun.com/exdb/mnist/
#
# In this problem you need to implement model that will learn to recognize
# handwritten digits
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


class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, params):
        super(SolutionModel, self).__init__()
        self.input_size = input_size
        self.params = params

        # Input channels = 1, output channels = 30
        self.conv1 = nn.Conv2d(1, 30, 5, 1)
        self.conv2 = nn.Conv2d(30, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop1(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop1(x)

        # Flatten
        x = x.view(-1, 4 * 4 * 50)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Solution():
    def __init__(self):
        self.lr = 0.0025
        self.momentum = 0.1

        # NOTE: Grids
        self.lr_grid = list(np.linspace(0.0009, 0.005, 10))
        self.momentum_grid = list(np.linspace(0.001, 1, 15))

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
        train_data = train_data.to(device)
        train_target = train_target.to(device)

        optimizer = optim.RMSprop(model.parameters(), lr=self.lr, alpha=0.9, momentum=0.1)
        batch_size = 256
        batches = train_data.size(0) // batch_size

        while True:
            ind = step % batches
            start_ind = batch_size * ind
            end_ind = start_ind + batch_size

            data = train_data[start_ind:end_ind]
            target = train_target[start_ind:end_ind]

            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            if time_left < 0.1:
                break

            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            output = model(data)
            # get the index of the max probability
            predict = output.max(1, keepdim=True)[1]
            # Number of correct predictions
            # Total number of needed predictions
            total = target.view(-1).size(0)
            # calculate loss
            loss = F.nll_loss(output, target)

            # if correct / total > 0.95:  # correct == total:
            #     # NOTE: Log stats
            #     stats = {
            #         'step': step,
            #         'loss': round(loss.item(), 5),
            #         'lr': self.lr,
            #         # 'hs': self.hidden_sizes,
            #         # 'act': self.activations,
            #         'mom': getattr(self, 'momentum', 0)
            #     }
            #     print(stats)
            #     if self.grid_search.enabled:
            #         results.append(stats)
            #     break

            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
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
        self.time_limit = 2
        self.size_limit = 1000000
        self.test_limit = 0.96


class DataProvider:
    def __init__(self):
        self.number_of_cases = 10
        print("Start data loading...")
        train_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=True, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
        test_dataset = torchvision.datasets.MNIST(
            './data/data_mnist', train=False, download=True,
            transform=torchvision.transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        self.train_data = next(iter(trainLoader))
        self.test_data = next(iter(test_loader))
        print("Data loaded")

    def select_data(self, data, digits):
        data, target = data
        data = data.to(device)
        target = target.to(device)
        mask = target == -1
        for digit in digits:
            mask |= target == digit
        indices = torch.arange(0, mask.size(0))[mask].long().to(device)
        return (torch.index_select(data, dim=0, index=indices), target[mask])

    def create_case_data(self, case):
        if case == 1:
            digits = [0, 1]
        elif case == 2:
            digits = [8, 9]
        else:
            digits = [i for i in range(10)]

        description = "Digits: "
        for ind, i in enumerate(digits):
            if ind > 0:
                description += ","
            description += str(i)
        train_data = self.select_data(self.train_data, digits)
        test_data = self.select_data(self.test_data, digits)
        return sm.CaseData(case, Limits(), train_data, test_data).set_description(description).set_output_size(10)

class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
# results = []

sm.SolutionManager(Config()).run(case_number=-1)
# if results:
#     with open('results.txt', 'w') as f:
#         text = '\n'.join(str(i) for i in sorted(
#             results, key=operator.itemgetter('step')))
#         f.write(text)
