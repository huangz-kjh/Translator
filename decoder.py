import torch
import math
from torch import nn
import torch.nn.functional as F
from sentence_tokenization import SentenceEmbedding

class LayerNormalization(nn.Module):
    
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.ones(parameters_shape))

    def forward(self, inputs):
        #  parameters_shape=[d_model]
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out
    
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAtention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, squence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, squence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, squence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out
    
class MultiHeadCrossAtention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, squence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, squence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, squence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, squence_length, d_model)
        out = self.linear_layer(values)
        return out
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAtention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAtention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden ,drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        residual_y = y
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.norm1(y + residual_y)

        residual_y = y
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.norm2(y + residual_y)

        residual_y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + residual_y)

        return y
    
# class SquentialDecoder(nn.Sequential):

#     def forward(self, *inputs):
#         x, y, mask = inputs
#         for module in self._modules.values():
#             y = module(x, y, mask)
#         return y
    
# class Decoder(nn.Module):
    
#     def __init__(self, d_model, ffn_hidden, num_heads, dropprob, num_layers=1):
#         super().__init__()
#         self.layers = SquentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, dropprob)
#                                          for _ in range(num_layers)])
        
#     def forward(self, x, y, mask):
#         y = self.layers(x, y, mask)
#         return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 language,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, language, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


if __name__=='__main__':
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    ffn_hidden = 2048
    num_layers = 5

    x = torch.randn( (batch_size, max_sequence_length, d_model) ) # English sentence positional encoded 
    y = torch.randn( (batch_size, max_sequence_length, d_model) ) # Kannada sentence positional encoded 
    mask = torch.full([max_sequence_length, max_sequence_length] , float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
    out = decoder(x, y, mask)