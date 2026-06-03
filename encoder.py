import torch
import math
from torch import nn
import torch.nn.functional as F

def sacled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MutliHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, max_sequence_length, self.d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = sacled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out 
    
class Normalnize(nn.Module):

    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.ones(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - var) / std
        out = self.gamma * y + self.beta
        return out

class PostitionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
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

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MutliHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = Normalnize(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PostitionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = Normalnize(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        residual_x = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)

        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, nums_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(nums_layers)])
        
    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    ffn_hidden = 2048
    num_layers = 5

    encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)

    x = torch.randn( (batch_size, max_sequence_length, d_model) ) # includes positional encoding
    out = encoder(x)

        
