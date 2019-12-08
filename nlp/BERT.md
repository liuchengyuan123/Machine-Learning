# BERT

### ELMO

ELMO是一种在词向量或词嵌入中表示词汇的新方法。

相比于其他的词嵌入模型，ELMO与它们最大的不同就是ELMO的每个词产生的词向量，在不同的语义下是不同的，与就是说ELMO的词向量是一个考虑全文的词向量。

### BERT

BERT模型，可以理解为把transformer的encoder单独拿出来。

从训练数据的角度讲，所有的文章（语义通顺，没有病句）都可以作为训练材料。

所谓预训练，指的是预训练出一个通用的语言模型，然后将它应用于具体场景时再精训练，为区别于针对某一语言的模型，我们称通用的语言模型为**语言表征模型**。

#### pre-training

怎么得到我们想要的语言表征模型呢？其实有很多种方法，其中最容易想到的就是盖住想要预测的词，从之前的语义中得到我们想要的目标。即
$$
p(y_t|y_1, y_2...y_{t-1})
$$
然而这种做法很明显只考虑了前面的词，生活中的预料很多情况下我们也需要考虑后面的词汇。

所以在这种做法的基础上我们需要再反向来一遍。**<font face=宋体>这里有没有觉得像ELMO呢</font>。**这就是bi-directional.

然而BERT的论文作者对这个模型仍然不满意，提出了deep bi-directional，称为上下文全向预测模型。作者建议使用transformer。

我们综合两种训练方法一起训练这个pre-training的模型。

- mask一个目标单词，让它预测这个单词是什么。
- 找两个句子，让模型判断这两个句子是否是连续的两句话。

经过这两种训练不同交替，得到的transformer参数就是期待的语言表征模型。
