import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_size)
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, seq_len: int, droput: float):
        super().__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(droput)

        # we need to create a matrix of shape (seq_len, embed_size) because for each position we need an embedding of size embed_size
        pe = torch.zeros(self.seq_len, self.embed_size)

        # create a position tensor of shape (seq_len, 1)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1) # numerator of the formula in sinusodial positional encoding
        # denominator of the formula in sinusodial positional encoding we do in log space for stability
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-math.log(10000.0) / self.embed_size)) 

        pe[:, 0::2] = torch.sin(position * div_term)  # apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cos to odd indices in the array\

        pe = pe.unsqueeze(0)  # add a batch dimension it becomes (1, seq_len, embed_size)

        # register as buffer so that it is not considered a model parameter but still gets saved and moved to GPU with the model
        self.register_buffer('pe', pe)  
        
    def forward(self, x):
        # x has the shape (batch_size, seq_len, embed_size)
        # add positional encoding to input embeddings and ensure no gradients are computed for pe as it is fixed 
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)  
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps # small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones((1,))) # scale parameter
        self.beta = nn.Parameter(torch.zeros((1,))) # shift parameter
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # compute mean over the last dimension i.e. embed_size
        std = x.std(dim=-1, keepdim=True) # compute std over the last dimension i.e. embed_size
        normalized_x = (x - mean) / (std + self.eps)
        return self.alpha * normalized_x + self.beta
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_size: int, ff_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden)
        self.fc2 = nn.Linear(ff_hidden, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_size % num_heads == 0 # ensure embed_size is divisible by num_heads
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads # dimension of each head
        '''
                    Input: (batch, seq_len, embed_size)
                                ↓
                        Split into num_heads heads
                                ↓
                    Each head: (batch, seq_len, embed_size/num_heads)
                                ↓
                    Attention computed per head
                                ↓
                        Concatenate heads
                                ↓
                    Output: (batch, seq_len, embed_size)
        '''
        self.Wq = nn.Linear(embed_size, embed_size)
        self.Wk = nn.Linear(embed_size, embed_size)
        self.Wv = nn.Linear(embed_size, embed_size)

        self.Wo = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        dk = query.shape[-1]

        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(dk)

        if mask is not None:
            attention_scores.masked_fill_(mask==0, 1e-9)
        
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        

    def forward(self, query, key, value, mask=None):
        Q = self.Wq(query)  # (batch_size, seq_len, embed_size) * (embed_size, embed_size) -> (batch_size, seq_len, embed_size)
        K = self.Wk(key)    # (batch_size, seq_len, embed_size) * (embed_size, embed_size) -> (batch_size, seq_len, embed_size)
        V = self.Wv(value) # (batch_size, seq_len, embed_size) * (embed_size, embed_size) -> (batch_size, seq_len, embed_size)

        # we keep the batch size and seq_len same and split the embed_size into num_heads and head_dim 
        # (batch_size, seq_len, embed_size) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim) 
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)  
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)      
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)  


        x, self.attention_scores = MultiHeadedAttention.attention(Q, K, V, mask, self.dropout)

        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, embed_size)
        x.transpose(1, 2).contiguous().view(x.size(0), -1, self.num_heads * self.head_dim)

        # (batch_size, seq_len, embed_size) -> (batch_size, seq_len, embed_size)
        return self.Wo(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadedAttention, feed_forward_block: FeedForwardNetwork, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadedAttention, cross_attention_block: MultiHeadedAttention, feed_forward_block: FeedForwardNetwork, dropout: float):

        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, enc_output, enc_mask, dec_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, dec_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, enc_output, enc_output, enc_mask))
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block)

        return x

class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, enc_output, enc_mask, dec_mask):
        for layer in self.layers:
            x = layer(x, enc_output, enc_mask, dec_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, embed_size: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        # (batch_size, seq_len, embed_size) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.fc(x), dim=-1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, encoder_embedding: InputEmbedding, decoder_embedding: InputEmbedding, encoder_positional_encoding: PositionalEncoding, decoder_positional_encoding: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_embedding = encoder_embedding
        self.dec_embedding = decoder_embedding
        self.enc_positional_encoding = encoder_positional_encoding
        self.dec_positional_encoding = decoder_positional_encoding
        self.projection_layer = projection_layer
    
    def encode(self, x, mask):
        x = self.enc_embedding(x)
        x = self.enc_positional_encoding(x)
        return self.encoder(x, mask)

    def decode(self, x, enc_output, enc_mask, dec_mask):
        x = self.dec_embedding(x)
        x = self.dec_positional_encoding(x)
        return self.decoder(x, enc_output, enc_mask, dec_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        src_seq_len: int, 
        tgt_seq_len: int, 
        embed_size: int = 512, 
        num_layers: int = 6,
        num_heads: int = 8, 
        ff_hidden: int = 2048,
        dropout: float = 0.1) -> Transformer:
    
    encoder_embedding = InputEmbedding(embed_size, src_vocab_size)
    decoder_embedding = InputEmbedding(embed_size, tgt_vocab_size)

    encoder_positional_encoding = PositionalEncoding(embed_size, src_seq_len, dropout)
    decoder_positional_encoding = PositionalEncoding(embed_size, tgt_seq_len, dropout)

    encoder_self_attention_blocks = nn.ModuleList([MultiHeadedAttention(embed_size, num_heads, dropout) for _ in range(num_layers)])
    decoder_self_attention_blocks = nn.ModuleList([MultiHeadedAttention(embed_size, num_heads, dropout) for _ in range(num_layers)])
    cross_attention_blocks = nn.ModuleList([MultiHeadedAttention(embed_size, num_heads, dropout) for _ in range(num_layers)])
    feed_forward_blocks = nn.ModuleList([FeedForwardNetwork(embed_size, ff_hidden, dropout) for _ in range(num_layers)])

    encoder_blocks = nn.ModuleList([EncoderBlock(encoder_self_attention_blocks[i], feed_forward_blocks[i], dropout) for i in range(num_layers)])
    decoder_blocks = nn.ModuleList([DecoderBlock(decoder_self_attention_blocks[i], cross_attention_blocks[i], feed_forward_blocks[i], dropout) for i in range(num_layers)])

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)

    projection_layer = ProjectionLayer(embed_size, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, encoder_embedding, decoder_embedding, encoder_positional_encoding, decoder_positional_encoding, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
