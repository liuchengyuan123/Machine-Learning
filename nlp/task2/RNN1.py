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

for filename in os.listdir('data/names/'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines('data/names/' + filename)
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

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# input = lineToTensor('Albert')
# hidden = torch.zeros(1, n_hidden)

# output, next_hidden = rnn(input[0], hidden)
# print(output)

def categoryFromOutput(output):
    top = torch.argmax(output[0], dim=0)
    return all_categories[top], top.item()

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

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

criterion = nn.NLLLoss()
lr = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-lr * p.grad.data)
    return output, loss.item()

n_iters = 70000
print_every = 5000
plot_every = 1000

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    if iter % print_every == 0:
        predict, predict_i = categoryFromOutput(output)
        correct = '✓' if predict == category else '✗ (%s)' % category
        print('%d %d%% %.4f %s / %s %s' % (iter, iter / n_iters * 100, loss, line, predict, correct))
        
'''
5000 5% 2.7751 Fujishima / Spanish ✗ (Japanese)
10000 10% 2.4353 Serafim / Arabic ✗ (Portuguese)
15000 15% 3.4000 Szwarc / Arabic ✗ (Polish)
20000 20% 2.8682 Spoerl / Dutch ✗ (Czech)
25000 25% 1.4448 Villa / Spanish ✓
30000 30% 0.7802 Wexler / German ✓
35000 35% 0.4153 Vuong / Vietnamese ✓
40000 40% 1.2937 Cleirich / Irish ✓
45000 45% 1.6546 Thibault / French ✓
55000 55% 0.6071 Hudecek / Czech ✓
60000 60% 0.2221 Brzezicki / Polish ✓
65000 65% 0.8045 Foerstner / German ✓
70000 70% 0.0602 Assaf / Arabic ✓
'''
