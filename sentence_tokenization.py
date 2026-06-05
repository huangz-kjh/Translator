import torch
import numpy as np
from collections import Counter
import re
from torch.utils.data import DataLoader, Dataset
from torch import nn

max_sequence_length = 200
NEG_INFTY = -1e9

english_file = '/home/huang/Transformer-Neural-Network/dataset/ch/en-zh/train_200k.en'
chinese_file = '/home/huang/Transformer-Neural-Network/dataset/ch/en-zh/train_200k.zh'

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
                      '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                      'Y', 'Z',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                      'y', 'z','?',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

chinese_vocabulary = [START_TOKEN, ' ', '！', '“', "”", '#', '$', '%', '&', "’", "‘", '（', '）', '*', '+', 
                    '，', '-', '。', '/',  
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '：', '<', '=', '>', '？', 
                    PADDING_TOKEN, END_TOKEN]

def build_zh_vocab(filepath, min_freq=1):
    counter_zh = Counter()

    with open(filepath, "r", encoding='utf-8') as f:
        for line in f:
            chars = list(line.strip())
            counter_zh.update(chars)

        word_to_id = {}

        for token in chinese_vocabulary:
            word_to_id[token] = len(word_to_id)

        for char, freq in counter_zh.items():
            if freq >= min_freq:
                word_to_id[char] = len(word_to_id)

        id_to_word = {v: k for k, v in word_to_id.items()}

    return list(counter_zh.keys()), word_to_id, id_to_word

def build_eng_vocab(filepath, min_freq=1):
    counter_en = Counter()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tokens = re.findall(r'[a-z]+|[0-9]+|[^\w\s]', line.lower())
            counter_en.update(tokens)
    
    word_to_id = {}

    for token in english_vocabulary:
        word_to_id[token] = len(word_to_id)

    for char, freq in counter_en.items():
        if freq >= min_freq:
            word_to_id[char] = len(word_to_id)

    id_to_word = {v: k for k, v in word_to_id.items()}

    return list(counter_en.keys()), word_to_id, id_to_word

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):  # 集合去重只要有不认识的字符直接宣布不行
        if token not in vocab:
            return False
    return True

def is_valid_zh_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)

def is_valid_eng_length(sentence, max_sequence_length):
    return len(re.findall(r"[a-z]+|[0-9]+|[^\w\s]", sentence.lower())) < (max_sequence_length - 1)

class TextDataset(Dataset):

    def __init__(self, english_sentences, chinese_sentences):
        super().__init__()
        self.english_sentences = english_sentences
        self.chinese_sentences = chinese_sentences
    
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, index):
        return self.english_sentences[index], self.chinese_sentences[index]

def tokenize_zh(sentence, language_to_index, start_token=True, end_token=True):
    sentence_word_indicies = [language_to_index[token] for token in list(sentence)]
    if start_token:
        sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
    if end_token:
        sentence_word_indicies.append(language_to_index[END_TOKEN,])
    for _ in range(len(sentence_word_indicies), max_sequence_length):
        sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
    return torch.tensor(sentence_word_indicies)

def tokenize_en(sentence, language_to_index, start_token=True, end_token=True):
    sentence_word_indicies = [language_to_index[token] for token in re.findall(r"[a-z]+|[0-9]+|[^\w\s]",
                                                                                sentence.lower())]
    if start_token:
        sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
    if end_token:
        sentence_word_indicies.append(language_to_index[END_TOKEN,])
    for _ in range(len(sentence_word_indicies), max_sequence_length):
        sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
    return torch.tensor(sentence_word_indicies)

def create_mask(eng_batch, zh_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    # diagonal=1 表示从主对角线偏移 1 的位置开始保留上三角
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, zh_sentence_length = len(eng_batch[idx]), len(zh_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        zh_chars_to_padding_mask = np.arange(zh_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, zh_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, zh_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, zh_chars_to_padding_mask, :] = True
    
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)

    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(0, max_sequence_length).reshape(max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, language, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.language = language
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token=True, end_token=True):
        
        def tokenize(sentence, start_token=True, end_token=True):
            if self.language == 'en':
                sentence_word_indicies = [self.language_to_index[token] for token in re.findall(r"[a-z]+|[0-9]+|[^\w\s]",sentence.lower())]
            else:
                sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[END_TOKEN])
            for _ in range(len(sentence_word_indicies), max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)
                    
        
        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token=True, end_token=True):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


if __name__=='__main__':
    english_vocabulary, english_to_index, index_to_english = build_eng_vocab(english_file)
    chinese_vocabulary, chinese_to_index, index_to_chinese = build_zh_vocab(chinese_file)

    # 1. 初始化
    english_embedding = SentenceEmbedding(
        max_sequence_length=200,
        d_model=512,
        language_to_index=english_to_index,
        language='en',
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PADDING_TOKEN=PADDING_TOKEN
    )

    chinese_embedding = SentenceEmbedding(
        max_sequence_length=200,
        d_model=512,
        language_to_index=chinese_to_index,
        language='zh',
        START_TOKEN=START_TOKEN,
        END_TOKEN=END_TOKEN,
        PADDING_TOKEN=PADDING_TOKEN
    )

    english_embedding = english_embedding.to(get_device())
    chinese_embedding = chinese_embedding.to(get_device())


    # 2. 准备输入数据
    with open(english_file, 'r') as file:
        english_sentences = file.readlines()
    with open(chinese_file, 'r') as file:
        chinese_sentences = file.readlines()

    # Limit Number of sentences ## 15886041
    #                                100000
    TOTAL_SENTENCES = 100000
    english_sentences = english_sentences[:TOTAL_SENTENCES]
    chinese_sentences = chinese_sentences[:TOTAL_SENTENCES]
    english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
    chinese_sentences = [sentence.rstrip('\n') for sentence in chinese_sentences]

    valid_sentence_indicies = []
    for index in range(len(chinese_sentences)):
        chinese_sentence, english_sentence = chinese_sentences[index], english_sentences[index]
        if is_valid_zh_length(chinese_sentence, max_sequence_length=max_sequence_length) \
            and is_valid_eng_length(english_sentence, max_sequence_length=max_sequence_length) \
            and is_valid_tokens(chinese_sentence, chinese_vocabulary):
                valid_sentence_indicies.append(index)
    
    chinese_sentences = [chinese_sentences[i] for i in valid_sentence_indicies]
    english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

    dataset = TextDataset(english_sentences, chinese_sentences)

    batch_size = 3 
    train_loader = DataLoader(dataset, batch_size)
    iterator = iter(train_loader)

    # 只取第一批次
    for batch_num, batch in enumerate(iterator):
        print(batch)
        if batch_num > 0:
            break

    # 3. 调用
    
    # eng_tokenized, zh_tokenized = [], []
    # for sentence_num in range(batch_size):
    #     eng_sentence, zh_sentence = batch[0][sentence_num], batch[1][sentence_num]
    #     zh_tokenized.append( tokenize_zh(zh_sentence, chinese_to_index, start_token=True, end_token=True) )
    #     eng_tokenized.append( tokenize_en(eng_sentence, english_to_index, start_token=False, end_token=False) )
    # eng_tokenized = torch.stack(eng_tokenized)
    # zh_tokenized = torch.stack(zh_tokenized)
    
    eng_embedded = english_embedding(batch[0])  # 形状: [batch_size, seq_len, d_model]
    zh_embedded = chinese_embedding(batch[1])   # 形状: [batch_size, seq_len, d_model]

    print(eng_embedded.shape)  # torch.Size([2, 200, 512])
    print(zh_embedded.shape)   # torch.Size([2, 200, 512])



    

