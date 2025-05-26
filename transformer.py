import torch
import torch.nn as nn
import math
import json
import os
from typing import Dict, List, Optional, Tuple, Union

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        # Self attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross attention
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class ContentStyle(nn.Module):
    def __init__(self, d_model, style_dim=64):
        super().__init__()
        self.style_encoder = nn.Sequential(
            nn.Linear(d_model, style_dim * 2),
            nn.LayerNorm(style_dim * 2),
            nn.ReLU(),
            nn.Linear(style_dim * 2, style_dim)
        )
        self.style_projector = nn.Linear(style_dim, d_model)
        
    def encode_style(self, x):
        # Average over sequence length to get style embedding
        avg_embedding = x.mean(dim=1)
        return self.style_encoder(avg_embedding)
        
    def apply_style(self, content, style):
        style_features = self.style_projector(style).unsqueeze(1)
        return content + style_features

class ControlCodeEmbedding(nn.Module):
    def __init__(self, num_codes, d_model):
        super().__init__()
        self.control_embeddings = nn.Embedding(num_codes, d_model)
        
    def forward(self, x, code_ids):
        code_embeddings = self.control_embeddings(code_ids).unsqueeze(1)
        return x + code_embeddings

class IndustryAdapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = residual + x
        return self.layer_norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_seq_length=5000,
                 num_control_codes=10, use_industry_adapters=False):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_decoder_layers)
        ])
        
        # Advanced features for content generation
        self.content_style = ContentStyle(d_model)
        self.control_code_embedding = ControlCodeEmbedding(num_control_codes, d_model)
        
        # Industry-specific adapters (optional)
        self.use_industry_adapters = use_industry_adapters
        if use_industry_adapters:
            self.industry_adapters = nn.ModuleDict({
                'ecommerce': IndustryAdapter(d_model),
                'finance': IndustryAdapter(d_model),
                'healthcare': IndustryAdapter(d_model),
                'technology': IndustryAdapter(d_model),
                'marketing': IndustryAdapter(d_model)
            })
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def apply_industry_adapter(self, x, industry):
        if not self.use_industry_adapters:
            return x
        
        if industry in self.industry_adapters:
            return self.industry_adapters[industry](x)
        return x
        
    def forward(self, src, tgt, control_codes=None, style_src=None, industry=None):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encoder
        src_embedded = self.dropout(self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model)))
        
        # Apply control codes if provided
        if control_codes is not None:
            src_embedded = self.control_code_embedding(src_embedded, control_codes)
        
        # Apply style conditioning if provided
        if style_src is not None:
            style_embedded = self.embedding(style_src) * math.sqrt(self.d_model)
            style_encoded = self.content_style.encode_style(style_embedded)
            src_embedded = self.content_style.apply_style(src_embedded, style_encoded)
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        # Apply industry-specific adapter if specified
        if industry is not None:
            enc_output = self.apply_industry_adapter(enc_output, industry)
            
        # Decoder
        tgt_embedded = self.dropout(self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # Apply industry-specific adapter to decoder output if specified
        if industry is not None:
            dec_output = self.apply_industry_adapter(dec_output, industry)
            
        output = self.fc_out(dec_output)
        return output
    
    def generate(self, src, max_length, control_codes=None, style_src=None, industry=None, 
                 temperature=1.0, top_k=50, top_p=0.9):
        """
        Generate text auto-regressively.
        
        Args:
            src: Input tensor with token ids
            max_length: Maximum sequence length to generate
            control_codes: Optional control codes for generation type
            style_src: Optional source text for style transfer
            industry: Optional industry specialization
            temperature: Sampling temperature (higher = more diverse)
            top_k: Number of top tokens to consider in sampling
            top_p: Cumulative probability threshold for nucleus sampling
        
        Returns:
            Generated token sequence
        """
        self.eval()
        with torch.no_grad():
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            
            # Encoder
            src_embedded = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))
            
            # Apply control codes if provided
            if control_codes is not None:
                src_embedded = self.control_code_embedding(src_embedded, control_codes)
            
            # Apply style conditioning if provided
            if style_src is not None:
                style_embedded = self.embedding(style_src) * math.sqrt(self.d_model)
                style_encoded = self.content_style.encode_style(style_embedded)
                src_embedded = self.content_style.apply_style(src_embedded, style_encoded)
                
            enc_output = src_embedded
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)
                
            # Apply industry-specific adapter if specified
            if industry is not None:
                enc_output = self.apply_industry_adapter(enc_output, industry)
            
            # Start with a tensor of just the BOS token
            ys = torch.ones(src.size(0), 1).fill_(1).long().to(src.device)
            
            for i in range(max_length-1):
                # Get target mask for the current output length
                tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(src.device)
                
                # Decoder forward pass
                tgt_embedded = self.pos_encoding(self.embedding(ys) * math.sqrt(self.d_model))
                dec_output = tgt_embedded
                for dec_layer in self.decoder_layers:
                    dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
                
                # Apply industry-specific adapter if specified
                if industry is not None:
                    dec_output = self.apply_industry_adapter(dec_output, industry)
                
                # Get prediction for next token
                out = self.fc_out(dec_output[:, -1])
                
                # Apply temperature
                out = out / temperature
                
                # Apply top-k sampling
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(out, min(top_k, out.size(-1)))
                    out = torch.zeros_like(out).scatter_(-1, top_k_indices, top_k_values)
                
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(out, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep first probs above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    out.scatter_(-1, indices_to_remove, float('-inf'))
                
                # Convert logits to probabilities
                probs = torch.softmax(out, dim=-1)
                
                # Sample from the distribution
                next_word = torch.multinomial(probs, 1)
                
                # Concatenate with previous output
                ys = torch.cat([ys, next_word], dim=1)
                
                # Stop if EOS token is generated
                if next_word.item() == 2:  # Assuming 2 is EOS token id
                    break
            
            return ys
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def save_with_config(self, path, tokenizer_info=None, control_code_map=None, industry_info=None):
        """Save model with configuration for easy loading"""
        # Save model weights
        torch.save(self.state_dict(), f"{path}.pt")
        
        # Save configuration information
        config = {
            "model_params": {
                "src_vocab_size": self.embedding.weight.size(0),
                "tgt_vocab_size": self.fc_out.weight.size(0),
                "d_model": self.d_model,
                "num_heads": self.encoder_layers[0].self_attn.num_heads,
                "num_encoder_layers": len(self.encoder_layers),
                "num_decoder_layers": len(self.decoder_layers),
                "d_ff": self.encoder_layers[0].feed_forward.linear1.out_features,
                "use_industry_adapters": self.use_industry_adapters,
            },
            "tokenizer_info": tokenizer_info,
            "control_codes": control_code_map,
            "industries": industry_info
        }
        
        with open(f"{path}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_with_config(cls, path):
        """Load model with configuration"""
        # Load configuration
        with open(f"{path}_config.json", 'r') as f:
            config = json.load(f)
        
        # Create model with saved parameters
        model = cls(**config["model_params"])
        
        # Load state dict
        model.load_state_dict(torch.load(f"{path}.pt"))
        
        return model, config 