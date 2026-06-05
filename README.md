# Transformer 神经网络 - 英中机器翻译

从零开始实现 Transformer 神经网络，用于英中机器翻译任务。

本项目提供了一个完整的 Transformer 架构实现，包括编码器、解码器、多头注意力、位置编码等核心组件。

![Transformer Architecture](Transformer_Architecture_complete.png)

## 项目概述

Transformer 是一种基于自注意力机制的深度学习模型，由 Google 在 2017 年提出。本项目完整实现了 Transformer 的所有关键组件，并训练了一个英中翻译模型。

## 主要特性

- **完整的 Transformer 实现**：从零构建的编码器、解码器、多头注意力机制
- **自注意力机制**：支持自注意力和交叉注意力
- **位置编码**：使用正弦/余弦函数实现位置编码
- **层归一化**：稳定的训练过程
- **掩码机制**：防止未来信息泄露和填充 token 干扰
- **英中翻译**：基于联合国平行语料库训练

## 文件结构

```
.
├── transformer.py              # Transformer 完整实现
├── encoder.py                  # 编码器组件
├── decoder.py                  # 解码器组件
├── sentence_tokenization.py    # 分词和句子嵌入
├── tf.py                       # Transformer 模型封装
├── translator.ipynb            # 训练和推理 notebook
├── Transformer_Trainer_Notebook.ipynb  # 完整训练流程
├── Self_Attention_for_Transformer_Neural_Networks.ipynb  # 自注意力机制教程
├── Mutlihead_Attention.ipynb   # 多头注意力教程
├── Layer_Normalization.ipynb   # 层归一化教程
├── Positional_Encoding_in_Transformer_neural_networks.ipynb  # 位置编码教程
├── Transformer_Encoder_EXPLAINED!.ipynb  # 编码器教程
├── Transformer_Decoder_EXPLAINED!.ipynb  # 解码器教程
├── Sentence_Tokenization.ipynb # 分词教程（英文）
├── Sentence_Tokenization_zh.ipynb  # 分词教程（中文）
├── dataset/                    # 训练数据集
│   ├── ch/                     # 中英平行语料
│   ├── english.txt             # 英文数据（卡纳达语）
│   └── kannada.txt             # 卡纳达语数据
└── requirements.txt            # Python 依赖
```

## 核心组件

### 1. Transformer 模型

完整的 Transformer 架构包含：

- **编码器 (Encoder)**：6 层，每层包含多头自注意力和前馈网络
- **解码器 (Decoder)**：6 层，每层包含多头自注意力、交叉注意力和前馈网络
- **句子嵌入 (Sentence Embedding)**：词嵌入 + 位置编码 + Dropout

### 2. 注意力机制

- **缩放点积注意力 (Scaled Dot-Product Attention)**
- **多头自注意力 (Multi-Head Self-Attention)**
- **多头交叉注意力 (Multi-Head Cross-Attention)**

### 3. 其他组件

- **位置编码 (Positional Encoding)**：正弦/余弦函数
- **层归一化 (Layer Normalization)**：可学习的参数 γ 和 β
- **前馈网络 (Positionwise Feed-Forward)**：两层全连接 + ReLU

## 环境配置

### 依赖要求

```bash
pip install torch numpy
```

### 设备支持

模型自动检测并使用 GPU（如果可用）：

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

## 使用方法

### 1. 初始化模型

```python
from tf import Transformer

# 模型参数
d_model = 512
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 6
max_sequence_length = 200

# 初始化 Transformer
transformer = Transformer(
    d_model=d_model,
    ffn_hidden=ffn_hidden,
    num_heads=num_heads,
    drop_prob=drop_prob,
    num_layers=num_layers,
    max_sequence_length=max_sequence_length,
    kn_vocab_size=zh_vocab_size,
    english_to_index=english_to_index,
    chinese_to_index=chinese_to_index,
    START_TOKEN='<START>',
    END_TOKEN='<END>',
    PADDING_TOKEN='<PADDING>'
)
```

### 2. 准备数据

```python
from sentence_tokenization import build_eng_vocab, build_zh_vocab

# 构建词汇表
english_vocabulary, english_to_index, index_to_english = build_eng_vocab('english.txt')
chinese_vocabulary, chinese_to_index, index_to_chinese = build_zh_vocab('chinese.txt')

# 加载句子对
with open('english.txt', 'r') as f:
    english_sentences = f.readlines()
with open('chinese.txt', 'r') as f:
    chinese_sentences = f.readlines()
```

### 3. 训练模型

```python
import torch
from torch import nn

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=chinese_to_index['<PADDING>'], reduction='none')
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

# 训练循环
for epoch in range(num_epochs):
    for batch_num, batch in enumerate(train_loader):
        eng_batch, zh_batch = batch
        
        # 创建掩码
        encoder_mask, decoder_mask, cross_mask = create_masks(eng_batch, zh_batch)
        
        # 前向传播
        zh_predictions = transformer(
            eng_batch, zh_batch,
            encoder_mask, decoder_mask, cross_mask,
            enc_start_token=False, enc_end_token=False,
            dec_start_token=True, dec_end_token=True
        )
        
        # 计算损失
        labels = transformer.decoder.sentence_embedding.batch_tokenize(zh_batch, start_token=False, end_token=True)
        loss = criterion(zh_predictions.view(-1, zh_vocab_size), labels.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4. 推理翻译

```python
def translate(sentence, transformer, max_length=max_sequence_length):
    zh_sentence = ("",)
    for word_counter in range(max_length):
        encoder_mask, decoder_mask, cross_mask = create_masks((sentence,), zh_sentence)
        
        predictions = transformer(
            (sentence,), zh_sentence,
            encoder_mask, decoder_mask, cross_mask,
            enc_start_token=False, enc_end_token=False,
            dec_start_token=True, dec_end_token=False
        )
        
        next_token_idx = torch.argmax(predictions[0][word_counter]).item()
        next_token = index_to_chinese[next_token_idx]
        zh_sentence = (zh_sentence[0] + next_token,)
        
        if next_token == '<END>':
            break
    
    return zh_sentence[0]

# 翻译示例
translation = translate("The use of remote sensing satellite data.", transformer)
```

## 数据集

本项目使用联合国平行语料库进行训练，数据位于 `dataset/ch/` 目录：

- `UNv1.0.en-zh.en` - 英文语料
- `UNv1.0.en-zh.zh` - 中文语料

数据预处理包括：

1. 过滤长度超过 `max_sequence_length` 的句子
2. 过滤包含未知 token 的句子
3. 构建词汇表映射

## 学习资源

本项目配套了详细的 YouTube 视频教程：

[**完整播放列表**](https://www.youtube.com/playlist?list=PLTl9hO2Oobd97qfWC40gOSU8C0iu0m2l4)

各组件详细讲解：
- [自注意力机制](Self_Attention_for_Transformer_Neural_Networks.ipynb)
- [多头注意力](Mutlihead_Attention.ipynb)
- [层归一化](Layer_Normalization.ipynb)
- [位置编码](Positional_Encoding_in_Transformer_neural_networks.ipynb)
- [编码器](Transformer_Encoder_EXPLAINED!.ipynb)
- [解码器](Transformer_Decoder_EXPLAINED!.ipynb)
- [分词（英文）](Sentence_Tokenization.ipynb)
- [分词（中文）](Sentence_Tokenization_zh.ipynb)

## 模型架构图

![Transformer Architecture](Transformer_Architecture_complete.png)

详细架构图文件：`Transformer_Architecture_complete.drawio`

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 作者

Ajay Halthor

## 致谢

本项目受 "Attention Is All You Need" 论文启发，实现了一个教学版本的 Transformer 模型。