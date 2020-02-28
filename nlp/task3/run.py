import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import unicodedata
import string

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)
def UnicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
    
category_lines = {}
all_categories = []
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [UnicodeToAscii(line) for line in lines]

for filename in os.listdir('../../data/names/'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines('../../data/names/' + filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# print(category_lines['Italian'][:5])

def letterToIndex(c):
    return all_letters.find(c)

def letterToTensor(letter):
    one_hot = torch.zeros(1, n_letters)
    one_hot[0][letterToIndex(letter)] = 1
    return one_hot

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        # print("input ", input.device, "hidden ", hidden.device)
        combined = torch.cat((input, hidden), 1)
        hidden = self.tanh(self.i2h(combined))
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()
lr = 0.005

optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)

# move to gpu
# rnn = rnn.cuda()
# criterion = criterion.cuda()

# input = lineToTensor('Albert')
# hidden = torch.zeros(1, n_hidden)

# output, next_hidden = rnn(input[0], hidden)
# print(output)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

# print(categoryFromOutput(output))

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line)


def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    line_tensor = line_tensor
    category_tensor = category_tensor
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

n_iters = 200000
print_every = 5000
plot_every = 1000

loss_sum = 0
correct_num = 0

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    predict, predict_i = categoryFromOutput(output)
    correct_num += (predict == category)
    loss_sum += loss
    if iter % print_every == 0:
        correct = '✓' if predict == category else '✗ (%s)' % category
        acc = correct_num / print_every
        print('%d %d%% %.4f %s / %s %s with accuracy %.4f%%' % (iter, iter / n_iters * 100, loss_sum / print_every, line, predict, correct, acc * 100))
        loss_sum = 0
        correct_num = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 50000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def OutPutProcess(totSpaces, totProcesses, curProcess):
    return "\r" + str(curProcess) + "/" + str(totProcesses) + "[" + (int(curProcess / totProcesses * totSpaces) * "#") + \
            (int(totSpaces - curProcess / totProcesses * totSpaces) * ".") + "]"

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1
    print(OutPutProcess(50, n_confusion, i + 1), end="")

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
