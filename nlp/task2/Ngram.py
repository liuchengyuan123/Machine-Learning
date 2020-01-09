import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = """When forty winters shall besiege thy brow,
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
trigrams = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(150):
    total_loss = 0
    correct_num = 0
    for context, target in trigrams:
        context_idx = torch.tensor([word_to_ix[word] for word in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idx)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if torch.argmax(log_probs, dim=1) == word_to_ix[target]:
            correct_num += 1
    if epoch % 10 == 0:
        print('epoch ====>', epoch)
        print('loss ====>', total_loss)
        print('accuracy ====> %.2f %%' % (correct_num / len(trigrams) * 100))
'''
epoch ====> 0
loss ====> 522.587329864502
accuracy ====> 0.88 %
epoch ====> 10
loss ====> 292.95620119571686
accuracy ====> 59.29 %
epoch ====> 20
loss ====> 81.14702689647675
accuracy ====> 94.69 %
epoch ====> 30
loss ====> 29.385152019560337
accuracy ====> 95.58 %
epoch ====> 40
loss ====> 17.695400096476078
accuracy ====> 96.46 %
epoch ====> 50
loss ====> 13.248517893254757
accuracy ====> 96.46 %       
epoch ====> 60
loss ====> 11.041018441319466
accuracy ====> 96.46 %       
epoch ====> 70
loss ====> 9.716049480251968
accuracy ====> 96.46 %      
epoch ====> 80
loss ====> 8.815495503135026
accuracy ====> 96.46 %
epoch ====> 90
loss ====> 8.149763791821897
accuracy ====> 96.46 %
epoch ====> 100
loss ====> 7.635357100050896
accuracy ====> 96.46 %
loss ====> 7.219624613877386
accuracy ====> 96.46 %
epoch ====> 120
loss ====> 6.875739000272006
accuracy ====> 96.46 %
epoch ====> 130
loss ====> 6.585319516947493
accuracy ====> 96.46 %
epoch ====> 140
loss ====> 6.337247955380008
accuracy ====> 96.46 %
'''
