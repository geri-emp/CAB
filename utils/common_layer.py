import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from utils import config


def gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def gen_inf_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if (config.USE_CUDA):
        return subsequent_mask.cuda()
    else:
        return subsequent_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        """
        :param input_depth: Size of last dimension of input(hidden_size)
        :param total_key_depth:
        :param total_value_depth:
        :param output_depth:Size last dimension of the final output
        :param num_heads:
        :param bias_mask:
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        # self.bias_mask = bias_mask
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, mask):
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads
        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)
        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        # Merge heads
        contexts = self._merge_heads(contexts)
        # Linear to get output
        outputs = self.output_linear(contexts)
        return outputs, attetion_weights


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        """
        :param input_depth: Size of last dimension of input(hidden_size)
        :param total_key_depth:
        :param total_value_depth:
        :param output_depth:Size last dimension of the final output
        :param num_heads:
        :param bias_mask:
        :param dropout:
        """
        super(DecoderMultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                  "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5  ## sqrt
        # self.bias_mask = bias_mask
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, mask):
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1) / self.num_heads
        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)
        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        # Merge heads
        contexts = self._merge_heads(contexts)
        # Linear to get output
        outputs = self.output_linear(contexts)
        return outputs, attetion_weights


class EmoAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, bias_mask=None, dropout=0.0):
        """
        :param input_depth: Size of last dimension of input(hidden_size)
        :param total_key_depth:
        :param total_value_depth:
        :param output_depth:Size last dimension of the final output
        :param bias_mask:
        :param dropout:
        """
        super(EmoAttention, self).__init__()
        self.query_scale = (total_key_depth) ** -0.5  ## sqrt
        # self.bias_mask = bias_mask
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth * 2, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        queries, emo_embedding, keys, values, mask = inputs
        # Do a linear for each component
        queries = self.query_linear(queries)
        queries = torch.cat((queries, emo_embedding), 2)
        keys = self.key_linear(keys)
        keys = torch.cat((keys, emo_embedding), 2)
        values = self.value_linear(values)
        values = torch.cat((values, emo_embedding), 2)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 2, 1))

        if mask is not None:
            logits = logits.masked_fill(mask, -1e18)

        ## attention weights
        attetion_weights = logits.sum(dim=1)
        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)
        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        # Linear to get output
        outputs = self.output_linear(contexts)
        return outputs, emo_embedding, outputs, outputs, mask


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        :param input_depth:
        :param filter_size:
        :param output_depth:
        :param layer_config: ll -> linear + ReLU + linear;cc -> conv + ReLU + conv etc.
        :param padding: left -> pad on the left side (to mask future data), both -> pad on both sides
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.1, attention_dropout=0.1, relu_dropout=0.1):
        """
        :param hidden_size:
        :param total_key_depth:
        :param total_value_depth:
        :param filter_size:
        :param num_heads:
        :param bias_mask: Masking tensor to prevent connections to future elements
        :param layer_dropout: Dropout for this layer
        :param attention_dropout:
        :param relu_dropout:
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        return y


class EncoderLastLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.1, attention_dropout=0.1, relu_dropout=0.1):
        """
        :param hidden_size:
        :param total_key_depth:
        :param total_value_depth:
        :param filter_size:
        :param num_heads:
        :param bias_mask: Masking tensor to prevent connections to future elements
        :param layer_dropout: Dropout for this layer
        :param attention_dropout:
        :param relu_dropout:
        """
        super(EncoderLastLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_own = LayerNorm(hidden_size)
        self.layer_norm_cross = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, inputs_x, mask_src=None, mask_x = None):
        x = inputs
        x_x = inputs_x
        # Layer Normalization
        # x_norm = self.layer_norm_mha(x)
        x_x_norm = self.layer_norm_own(x_x)
        # Multi-head attention
        y, _ = self.multi_head_attention(x_x_norm, x_x_norm, x_x_norm, mask_x)
        #dropout after self-attention
        x_x = self.dropout(x_x + y)
        #layer norm before cross-attention
        x_x_norm = self.layer_norm_cross(x_x)
        # Multi-head attention
        y, _ = self.multi_head_attention(x_x_norm, x, x, mask_src)

        # Dropout and residual
        x_x = self.dropout(x_x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x_x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x_x + y)

        return x, y, mask_src, mask_x


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, vocab_size=None, layer_dropout=0, attention_dropout=0.1, relu_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = DecoderMultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                                      hidden_size, num_heads, None, attention_dropout)

        self.multi_head_attention_knowledge = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """
        x, encoder_outputs, attention_weight, knowledge_attention, mask = inputs  # x, encoder_output, [], (mask_src,dec_mask)
        mask_src, dec_mask = mask

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        # Dropout and after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        if len(knowledge_attention) != 0:
            # Multi-head knowledge attention
            knowledge_encoder_outputs = torch.mul(knowledge_attention.unsqueeze(2), encoder_outputs)
            y, _ = self.multi_head_attention_knowledge(x_norm, knowledge_encoder_outputs, knowledge_encoder_outputs, mask_src)

            # Dropout and residual after knowledge attention
            x = self.dropout(x + y)

            # Layer Normalization
            x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # y = self.layer_norm_end(y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs, attention_weight, knowledge_attention, mask


class TransEncoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]

    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, use_mask=False, universal=False):
        """
        :param embedding_size:
        :param hidden_size:
        :param num_layers:
        :param num_heads:
        :param total_key_depth: Must be divisible by num_head
        :param total_value_depth: Must be divisible by num_head
        :param filter_size: Hidden size of the middle layer in FFN
        :param max_length: Max sequence length (required for timing signal)
        :param input_dropout: Dropout just after embedding
        :param layer_dropout: Dropout for each layer
        :param attention_dropout: Dropout probability after attention (Should be non-zero only during training)
        :param relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        :param use_mask: Set to True to turn on future value masking
        """
        super(TransEncoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = gen_timing_signal(max_length, hidden_size)  # torch.Size([1,max_length,hidden_size])
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            ## for t
            self.position_signal = gen_timing_signal(num_layers, hidden_size)
        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            gen_inf_mask(max_length) if use_mask else None,  # torch.Size([1, 1, max_length, max_length])
            layer_dropout,
            attention_dropout,
            relu_dropout
        )
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        self.last_layer = nn.ModuleList([EncoderLastLayer(*params) for _ in range(num_layers)])  # 双多头注意力会调用两次这个层
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        if (config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, inputs_x, mask_src, mask_x):
        # input:[batch,length,emb_dim) mask:[batch,1,length]
        # Add input dropout           input:utterance embedding+speaker embedding
        x = self.input_dropout(inputs)
        # Project to hidden size
        x = self.embedding_proj(x)
        if (self.universal):
            if (config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask_src)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)  # splice in torch.Size([1, 1000, 300])
            inputs_x += self.timing_signal[:, :inputs_x.shape[1], :].type_as(
                inputs_x.data)   # splice in torch.Size([1, 1000, 300])
            # inputs_x += self.timing_signal[:, :inputs_x.shape[1], :].type_as(inputs_x.data) # splice in torch.Size([1, 1000, 300])

            for i in range(self.num_layers):
                x = self.enc[i](x, mask_src)
            for i in range(self.num_layers):
                x, inputs_x, _, _ = self.last_layer[i](x, inputs_x, mask_src = mask_src, mask_x = mask_x)
            # x = self.last_layer(x, inputs_x, mask_src = mask_src, mask_x = mask_x)
            # x_l = self.last_layer(x,inputs_l,mask)
            y = self.layer_norm(inputs_x)
            # y_l = self.layer_norm(x_l)

        return y


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, use_mask=False, universal=False):
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = gen_timing_signal(max_length, hidden_size)  # torch.Size([1, 1000, 300])

        if (self.universal):
            ## for t
            self.position_signal = gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  gen_inf_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)  # (300,300)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

        if (config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):  # input:[batch,length,emb_dim) mask:[batch,1,length]
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)

        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=200, input_dropout=0, layer_dropout=0,
                 attention_dropout=0.1, relu_dropout=0.1, universal=False):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = gen_timing_signal(max_length, hidden_size)

        self.mask = get_attn_subsequent_mask(max_length)
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  gen_inf_mask(max_length),  # mandatory
                  None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, k_scores, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Run decoder attn_dist is the attention of enc_dec
        y, _, attn_dist, _, _ = self.dec((x, encoder_output, [], k_scores, (mask_src, dec_mask)))
        # Final layer normalization
        y = self.layer_norm(y)
        return y, attn_dist


class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()

        step = 0
        # for l in range(self.num_layers):
        while (((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if (decoding):
                state, _, attention_weight = fn((state, encoder_output, []))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = (
                        (state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            if (decoding):
                if (step == 0):  previous_att_weight = torch.zeros_like(attention_weight).cuda()  ## [B, S, src_size]
                previous_att_weight = ((attention_weight * update_weights.unsqueeze(-1)) + (
                            previous_att_weight * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step += 1

        if (decoding):
            return previous_state, previous_att_weight, (remainders, n_updates)
        else:
            return previous_state, (remainders, n_updates)

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def get_input_from_batch(batch):
    enc_batch = batch['input_batch']
    enc_lens = batch['input_lengths']
    batch_size, max_enc_len = enc_batch.size()
    assert len(enc_lens) == batch_size
    enc_padding_mask = sequence_mask(enc_lens, max_len = max_enc_len).float()
    if config.USE_CUDA:
        enc_padding_mask = enc_padding_mask.cuda()
    return enc_batch

class Beam:
    """ Beam search """

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [
            torch.full((size,), config.PAD_idx, dtype=torch.long, device=device)
        ]
        self.next_ys[0][0] = config.SOS_idx

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True
        )  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True
        )  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = torch.div(best_scores_id, num_words, rounding_mode= 'floor')
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == config.EOS_idx:
            # self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[config.SOS_idx] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

class Translator(object):
    """ Load with trained model and handle the beam search """

    def __init__(self, model, lang):

        self.model = model
        self.lang = lang
        self.vocab_size = lang.n_words
        self.beam_size = config.beam_size
        self.device = 'cuda' if config.USE_CUDA else 'cpu'

    def beam_search(self,batch, ty, max_dec_step):
        """ Translation work in one batch """

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """ Indicate the position of an instance in a tensor. """
            return {
                inst_idx: tensor_position
                for tensor_position, inst_idx in enumerate(inst_idx_list)
            }

        def collect_active_part(
            beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm
        ):
            """ Collect tensor parts associated to active instances. """

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
            src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list
        ):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [
                inst_idx_to_position_map[k] for k in active_inst_idx_list
            ]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(
                src_seq, active_inst_idx, n_prev_active_inst, n_bm
            )
            active_src_enc = collect_active_part(
                src_enc, active_inst_idx, n_prev_active_inst, n_bm
            )

            active_encoder_db = None

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            return (
                active_src_seq,
                active_encoder_db,
                active_src_enc,
                active_inst_idx_to_position_map,
            )

        def beam_decode_step(
            inst_dec_beams,
            len_dec_seq,
            src_seq,
            enc_output,
            inst_idx_to_position_map,
            n_bm,
            enc_batch_extend_vocab,
            extra_zeros,
            mask_src,
            encoder_db,
            mask_transformer_db,
            DB_ext_vocab_batch,
        ):
            """ Decode and update beam status, and then return active beam idx """

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [
                    b.get_current_state() for b in inst_dec_beams if not b.done
                ]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(
                    1, len_dec_seq + 1, dtype=torch.long, device=self.device
                )
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(
                    n_active_inst * n_bm, 1
                )
                return dec_partial_pos

            def predict_word(
                dec_seq,
                dec_pos,
                src_seq,
                enc_output,
                n_active_inst,
                n_bm,
                enc_batch_extend_vocab,
                extra_zeros,
                mask_src,
                encoder_db,
                mask_transformer_db,
                DB_ext_vocab_batch,
            ):
                ## masking
                mask_trg = dec_seq.data.eq(config.PAD_idx).unsqueeze(1)
                mask_src = torch.cat([mask_src[0].unsqueeze(0)] * mask_trg.size(0), 0)
                input_vector = self.model.embedding(dec_seq)
                input_vector_start = torch.cat((input_vector[:, 0], z_s.repeat(5,1), z_l.repeat(5,1)), dim=1)
                input_vector_start = self.model.trans_input(input_vector_start).unsqueeze(1)
                input_vector = torch.cat((input_vector_start, input_vector[:, 1:]), dim=1)

                dec_output, attn_dist = self.model.decoder(
                    input_vector, enc_output, (mask_src, mask_trg)
                )

                db_dist = None

                prob = self.model.generator(
                    dec_output,
                    attn_dist,
                    enc_batch_extend_vocab,
                    extra_zeros,
                    True,
                    1
                )
                # prob = F.log_softmax(prob,dim=-1) #fix the name later
                word_prob = prob[:, -1]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(
                inst_beams, word_prob, inst_idx_to_position_map
            ):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(
                        word_prob[inst_position]
                    )
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(
                dec_seq,
                dec_pos,
                src_seq,
                enc_output,
                n_active_inst,
                n_bm,
                enc_batch_extend_vocab,
                extra_zeros,
                mask_src,
                encoder_db,
                mask_transformer_db,
                DB_ext_vocab_batch,
            )

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map
            )

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [
                    inst_dec_beams[inst_idx].get_hypothesis(i)
                    for i in tail_idxs[:n_best]
                ]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            enc_batch, dec_batch, enc_batch_extend_vocab, extra_zeros, dec_ext_batch, \
            mask_src, mask_s, mask_l, s_encoder_outputs, l_encoder_outputs, \
            r_encoder_outputs, act_encoder_outputs, act_embedding = self.model.forward(batch, ty)

            # torch.Size([32, 32])
            emotion_s_logit_prob = self.model.emo_s_fc(s_encoder_outputs[:, 0])
            softmax_s_logit = F.log_softmax(emotion_s_logit_prob, dim=-1)
            pred_emotion_s_embedding = torch.matmul(softmax_s_logit, self.model.emo_embedding.weight)
            s_encoder_outputs = self.model.emo_attention(s_encoder_outputs, pred_emotion_s_embedding.unsqueeze(1).expand(
                pred_emotion_s_embedding.size(0), s_encoder_outputs.size(1), pred_emotion_s_embedding.size(1)),
                                                   s_encoder_outputs, s_encoder_outputs, mask_s)

            emotion_l_logit_prob = self.model.emo_l_fc(l_encoder_outputs[:, 0])
            softmax_l_logit = F.log_softmax(emotion_l_logit_prob, dim=-1)
            pred_emotion_l_embedding = torch.matmul(softmax_l_logit, self.model.emo_embedding.weight)
            l_encoder_outputs = self.model.emo_attention(l_encoder_outputs, pred_emotion_l_embedding.unsqueeze(1).expand(
                pred_emotion_l_embedding.size(0), l_encoder_outputs.size(1), pred_emotion_l_embedding.size(1)),
                                                   l_encoder_outputs, l_encoder_outputs, mask_l)

            emotion_s_kld_loss, z_s = self.model.emo_s_latent_layer(s_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train=False)
            emotion_l_kld_loss, z_l = self.model.emo_l_latent_layer(l_encoder_outputs[:, 0], r_encoder_outputs[:, 0], train=False)
            act_kld_loss, z_a = self.model.act_latent_layer(act_encoder_outputs[:, 0], r_encoder_outputs[:, 0], act_embedding, train=False)
            # (batch_size,600)
            # gen_inputs = act_encoder_outputs[:, 0] + z_a
            gen_inputs = torch.cat([act_encoder_outputs[:, 0], z_a], 1)
            act_logit_prob = self.model.act_fc(gen_inputs)
            act_logit_prob = F.log_softmax(act_logit_prob, dim=-1)
            pred_act_embedding = torch.matmul(act_logit_prob, self.model.act_embedding.weight)
            selected_attribute_embedding = pred_act_embedding

            # dec_inputs = torch.cat([gen_inputs, selected_attribute_embedding], 1)
            # dec_inputs = self.model.trans_output(dec_inputs).unsqueeze(1)
            # act_encoder_outputs = torch.cat((dec_inputs, act_encoder_outputs[:, 1:]),dim = 1)

            src_enc = act_encoder_outputs
            encoder_db = None

            mask_transformer_db = None
            DB_ext_vocab_batch = None

            # -- Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.size()
            src_seq = enc_batch.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
                active_inst_idx_list
            )

            # -- Decode
            for len_dec_seq in range(1, max_dec_step + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams,
                    len_dec_seq,
                    src_seq,
                    src_enc,
                    inst_idx_to_position_map,
                    n_bm,
                    enc_batch_extend_vocab,
                    extra_zeros,
                    mask_src,
                    encoder_db,
                    mask_transformer_db,
                    DB_ext_vocab_batch,
                )

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (
                    src_seq,
                    encoder_db,
                    src_enc,
                    inst_idx_to_position_map,
                ) = collate_active_info(
                    src_seq,
                    encoder_db,
                    src_enc,
                    inst_idx_to_position_map,
                    active_inst_idx_list,
                )

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        ret_sentences = []
        for d in batch_hyp:
            ret_sentences.append(
                " ".join([self.model.vocab.index2word[idx] for idx in d[0]])
                )


        return ret_sentences  # , batch_scores

def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

