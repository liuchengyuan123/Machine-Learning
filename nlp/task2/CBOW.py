import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CONTEXT_SIZE = 2
EMBEDDING_SIZE = 10

raw_text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_idx = {word: ix for ix, word in enumerate(vocab)}
data = []

def get_idx(pos):
    return word_to_idx[raw_text[pos]]

for center_word_pos in range(2, len(raw_text) - 2):
    context_words = []
    for w in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
        if w is 0:
            continue
        context_words.append(get_idx(center_word_pos + w))
    data.append((context_words, get_idx(center_word_pos)))

# print(data[: 5])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.linear1 = nn.Linear(EMBEDDING_SIZE, 128)
        self.linear2 = nn.Linear(128 * 2 * CONTEXT_SIZE, vocab_size)
    def forward(self, x):
        em = self.embed(x)
        z1 = F.relu(self.linear1(em))
        y2 = self.linear2(z1.view(1, -1))
        z2 = F.log_softmax(y2, dim=1)
        return z2

model = Model()
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

NUM_EPOCH = 20
for epoch in range(NUM_EPOCH):
    total_loss = 0
    correct_num = 0
    for (context, target_word) in data:
        context_words = torch.tensor(np.array(context), dtype=torch.long)
        target = torch.tensor(np.array([target_word]), dtype=torch.long)
        y_pred = model.forward(context_words)
        loss = criterion(y_pred, target)
        total_loss += loss.item()

        if target_word == torch.argmax(y_pred, dim=1):
            correct_num += 1
        model.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch ====>', epoch)
    print('loss ====>', total_loss)
    print('accuracy ====> %.2f %%' % (correct_num / len(data) * 100))
'''
epoch ====> 0
loss ====> 586.6309773921967
accuracy ====> 1.80 %
epoch ====> 1
loss ====> 344.53840285912156
accuracy ====> 23.42 %
epoch ====> 2
loss ====> 14.4143593905319
accuracy ====> 98.20 %
epoch ====> 3
loss ====> 0.2250337263310982
accuracy ====> 100.00 %
epoch ====> 4
loss ====> 0.17018777722062595
accuracy ====> 100.00 %
epoch ====> 5
loss ====> 0.1350790977534686
accuracy ====> 100.00 %
epoch ====> 6
loss ====> 0.10998713205747634
accuracy ====> 100.00 %
epoch ====> 7
loss ====> 0.09125963579543139
accuracy ====> 100.00 %
epoch ====> 8
loss ====> 0.0768284556813228
accuracy ====> 100.00 %
epoch ====> 9
loss ====> 0.0654590881155741
accuracy ====> 100.00 %
epoch ====> 10
loss ====> 0.056352553725162124
accuracy ====> 100.00 %
epoch ====> 11
loss ====> 0.048954822222242456
accuracy ====> 100.00 %
epoch ====> 12
loss ====> 0.042855443040423324
accuracy ====> 100.00 %
epoch ====> 13
loss ====> 0.037742167798398896
accuracy ====> 100.00 %
epoch ====> 14
loss ====> 0.03343640804220627
accuracy ====> 100.00 %
epoch ====> 15
loss ====> 0.0297745810063077
accuracy ====> 100.00 %
epoch ====> 16
loss ====> 0.026632091052732676
accuracy ====> 100.00 %
epoch ====> 17
loss ====> 0.023913001814520385
accuracy ====> 100.00 %
epoch ====> 18
loss ====> 0.021546129296410754
accuracy ====> 100.00 %
epoch ====> 19
loss ====> 0.01947861403343154
accuracy ====> 100.00 %
'''
