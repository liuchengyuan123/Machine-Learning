# seq2seq module

### 概述

seq2seq模型的输入为一个sequence，输出也是一个sequence，然而，这两个sequence的长度不一定相等。

seq2seq为一个encoder-decoder模型，输入序列在encoder里编码成固定长度的向量，输出为解码后不固定长的向量。
$$
x = \left \{x_1, x_2, x_3...x_{n_x} \right \} \\y = \left \{ y_1, y_2, y_3...y_{n_y} \right \}
$$
seq2seq模型的优化函数为
$$
p(y|x) = \prod_{t=1}^{n_y}p(y_t|y_1, y_2...y_{t-1},x)
$$
即输出和之前所有的输出、输入的整个句子有关。

#### 缺点

然而该模型存在数值下溢（numerical underflow）的问题，每一项概率都很小，到最后可能会无法计算，所以一般使用对数来量化该函数。
$$
P(y|x) = \sum_{t=1}^{n_y}\log P(y_t|y_1, y_2...y_{t - 1}, x)
$$

### base seq2seq

输入序列先在encoder中完成编码，这个阶段称为编码阶段，可以是RNN、CNN等结构，最后时刻的输出编码即为中间语义。

然后将表示语义的向量作为decoder的输入，进行解码，这个阶段称为解码阶段。一直预测，直到decoder某个时刻的输出为<END>。

### 加入attention机制

LSTM虽然有记忆性，但是如果输入句子很长，可想而知，一个固定长度的向量很难完整表达我们输入的信息，或者说，当两个有关系的词相距很远，可能不能很好的顾及前面的信息。这是Attention机制就被提出，用来解决这个问题。

而attention机制的结构比较复杂，目前我还没有理解为什么这样做是对的。

假设$s_i$为当前$RNN$单元的隐藏层状态，$y_i$为前一次的输出，那么$s_i = f(s_{i-1}, y_i, c_i)$，其中
$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$
即encoder阶段的每个时刻的输出状态的加权和。
$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{Tx} \exp(e_{ik})}\\e_{ij} = v^{T} \tanh(W_as_{i-1} + U_a h_j)
$$

### 预处理

**embedding降维**

