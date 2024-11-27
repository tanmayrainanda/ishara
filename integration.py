import os
import json
import random
import math
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import Levenshtein
from tqdm.auto import tqdm

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 3, output_dim: int = 52):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Linear(64 * input_channels, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 384):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        sinusoid = torch.einsum('i,j->ij', position, inv_freq)
        self.register_buffer('sin', sinusoid.sin())
        self.register_buffer('cos', sinusoid.cos())

    def forward(self, x):
        sin = self.sin[:x.shape[1]]
        cos = self.cos[:x.shape[1]]
        return sin, cos

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        q_rot, q_pass = q[..., :self.head_dim//2], q[..., self.head_dim//2:]
        k_rot, k_pass = k[..., :self.head_dim//2], k[..., self.head_dim//2:]
        
        # Rotate q and k
        q = torch.cat([-q_rot * sin[:, None] + q_rot * cos[:, None], q_pass], dim=-1)
        k = torch.cat([-k_rot * sin[:, None] + k_rot * cos[:, None], k_pass], dim=-1)
        
        # Reshape for attention
        q = q.transpose(1, 2)  # (B, H, L, D/H)
        k = k.transpose(1, 2)  # (B, H, L, D/H)
        v = v.transpose(1, 2)  # (B, H, L, D/H)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # (B, H, L, D/H)
        out = out.transpose(1, 2).contiguous()  # (B, L, H, D/H)
        out = out.reshape(B, L, D)  # (B, L, D)
        
        return self.out_proj(out)

class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadAttention(dim, num_heads, dropout)
        
        # Convolution module
        self.conv_module = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        # Feed forward module
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self attention
        residual = x
        x = self.norm1(x)
        x = self.mhsa(x, sin, cos, mask)
        x = self.dropout(x)
        x = residual + x * self.scale
        
        # Convolution module
        residual = x
        x = self.norm2(x)
        x = self.conv_module(x.transpose(1, 2)).transpose(1, 2)
        x = residual + x * self.scale
        
        # Feed forward
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x * self.scale
        
        return x

class SqueezeformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadAttention(dim, num_heads, dropout)
        
        # Feed forward module
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        
        # Convolution module
        self.conv_module = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First feed forward
        residual = x
        x = self.norm1(x)
        x = self.ff1(x)
        x = residual + x * self.scale
        
        # Self attention
        residual = x
        x = self.norm2(x)
        x = self.mhsa(x, sin, cos, mask)
        x = self.dropout(x)
        x = residual + x * self.scale
        
        # Convolution module
        residual = x
        x = self.norm3(x)
        x = self.conv_module(x.transpose(1, 2)).transpose(1, 2)
        x = residual + x * self.scale
        
        # Second feed forward
        residual = x
        x = self.norm4(x)
        x = self.ff2(x)
        x = residual + x * self.scale
        
        return x

class ASLTranslationModel(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 130,
        feature_dim: int = 208,
        num_classes: int = 59,
        num_layers: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature extractors for different landmark types
        self.face_extractor = FeatureExtractor(3, 52)
        self.pose_extractor = FeatureExtractor(3, 52)
        self.left_hand_extractor = FeatureExtractor(3, 52)
        self.right_hand_extractor = FeatureExtractor(3, 52)
        self.all_landmarks_extractor = FeatureExtractor(3, 208)
        
        # Positional embedding
        self.rotary_pe = RotaryPositionalEmbedding(feature_dim)
        
        # Parallel encoders
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(feature_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.squeezeformer_layers = nn.ModuleList([
            SqueezeformerBlock(feature_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Output layers
        self.confidence_head = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, L, C = x.shape  # batch, time, landmarks, channels
        
        # Split landmarks into types
        face = x[:, :, :76]  # first 76 landmarks are face
        pose = x[:, :, 76:88]  # next 12 landmarks are pose
        left_hand = x[:, :, 88:109]  # next 21 landmarks are left hand
        right_hand = x[:, :, 109:]  # remaining 21 landmarks are right hand
        
        # Extract features for each type
        face_feats = self.face_extractor(face)
        pose_feats = self.pose_extractor(pose)
        left_hand_feats = self.left_hand_extractor(left_hand)
        right_hand_feats = self.right_hand_extractor(right_hand)
        
        # Concatenate all type-specific features
        type_feats = torch.cat(
            [face_feats, pose_feats, left_hand_feats, right_hand_feats],
            dim=-1
        )
        
        # Get features for all landmarks together
        all_feats = self.all_landmarks_extractor(x)
        
        # Combine both feature sets
        features = torch.cat([all_feats, type_feats], dim=-1)
        
        # Get rotary embeddings
        sin, cos = self.rotary_pe(features)
        
        # Pass through parallel encoder layers
        conformer_out = features
        squeezeformer_out = features
        
        for conf_layer, squeeze_layer in zip(
            self.conformer_layers,
            self.squeezeformer_layers
        ):
            conformer_out = conf_layer(conformer_out, sin, cos, mask)
            squeezeformer_out = squeeze_layer(squeezeformer_out, sin, cos, mask)
        
        # Combine encoder outputs
        encoder_out = conformer_out + squeezeformer_out
        
        # Get confidence score from first token
        confidence = self.confidence_head(encoder_out[:, 0]).squeeze(-1)
        
        # Decode if target is provided (training) else return encoder output
        if tgt is not None:
            decoder_out = self.decoder(
                tgt,
                encoder_out,
                tgt_mask=generate_square_subsequent_mask(tgt.size(1)).to(tgt.device),
                memory_mask=mask
            )
            output = self.classifier(decoder_out)
        else:
            output = self.classifier(encoder_out)
        
        return output, confidence

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class ASLTokenizer:
    """Tokenizer for ASL fingerspelling sequences"""
    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r') as f:
            self.char_to_idx = json.load(f)
            
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to token indices"""
        tokens = [self.sos_token]
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.pad_token))
        tokens.append(self.eos_token)
        return torch.tensor(tokens)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Convert token indices to text"""
        text = []
        for token in tokens:
            if token.item() == self.eos_token:
                break
            if token.item() not in [self.pad_token, self.sos_token]:
                text.append(self.idx_to_char[token.item()])
        return ''.join(text)

class ASLDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        tokenizer: ASLTokenizer,
        max_len: int = 384,
        augment: bool = True,
        fold: int = 0,
        num_folds: int = 4,
        mode: str = 'train'
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        # Split by participant_id for cross-validation
        participants = df['participant_id'].unique()
        np.random.seed(42)
        np.random.shuffle(participants)
        fold_size = len(participants) // num_folds
        val_participants = participants[fold * fold_size:(fold + 1) * fold_size]
        
        if mode == 'train':
            self.df = df[~df['participant_id'].isin(val_participants)]
        else:
            self.df = df[df['participant_id'].isin(val_participants)]
            
        # Preload all data into memory for faster training
        self.data = {}
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Loading data'):
            file_path = self.data_dir / f"{row['sequence_id']}.npy"
            self.data[row['sequence_id']] = np.load(file_path)
            
    def normalize_landmarks(self, landmarks: np.ndarray) -> torch.Tensor:
        """Normalize landmarks with mean and std"""
        # Convert to tensor
        landmarks = torch.from_numpy(landmarks).float()
        
        # Calculate mean and std per dimension
        mean = landmarks.mean(dim=0, keepdim=True)
        std = landmarks.std(dim=0, keepdim=True)
        std[std == 0] = 1
        
        # Normalize
        landmarks = (landmarks - mean) / std
        
        # Fill nans with zeros
        landmarks = torch.nan_to_num(landmarks)
        
        return landmarks
    
    def augment_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to landmark sequence"""
        # Time augmentations
        if random.random() < 0.8:
            # Random resize along time axis
            scale = random.uniform(0.8, 1.2)
            T = landmarks.shape[0]
            new_T = int(T * scale)
            landmarks = F.interpolate(
                landmarks.permute(2, 0, 1)[None],
                size=new_T,
                mode='linear',
                align_corners=False
            )[0].permute(1, 2, 0)
            
        if random.random() < 0.5:
            # Random time shift
            shift = random.randint(-10, 10)
            landmarks = torch.roll(landmarks, shift, dims=0)
            
        # Spatial augmentations
        if random.random() < 0.8:
            # Random spatial affine
            angle = random.uniform(-30, 30)
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-0.2, 0.2)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            
            theta = torch.tensor([
                [scale * math.cos(angle), -scale * math.sin(angle) + shear, translate[0]],
                [scale * math.sin(angle) + shear, scale * math.cos(angle), translate[1]]
            ]).float()
            
            grid = F.affine_grid(
                theta[None],
                size=(1, 1, landmarks.shape[1], 2),
                align_corners=False
            )
            landmarks_2d = landmarks[..., :2].reshape(-1, landmarks.shape[1], 2)
            landmarks_2d = F.grid_sample(
                landmarks_2d[:, None],
                grid,
                align_corners=False
            )[:, 0]
            landmarks[..., :2] = landmarks_2d.reshape(landmarks.shape[:-1] + (2,))
        
        # Landmark dropping
        if random.random() < 0.5:
            # Randomly drop fingers
            num_fingers = random.randint(2, 6)
            num_windows = random.randint(2, 3)
            
            for _ in range(num_windows):
                window_start = random.randint(0, landmarks.shape[0] - 20)
                window_size = random.randint(10, 20)
                window_end = min(window_start + window_size, landmarks.shape[0])
                
                for _ in range(num_fingers):
                    finger_start = random.randint(88, 130 - 4)  # Hand landmarks
                    landmarks[window_start:window_end, finger_start:finger_start+4] = 0
                    
        if random.random() < 0.5:
            # Drop face or pose landmarks
            if random.random() < 0.5:
                landmarks[:, :76] = 0  # Face
            else:
                landmarks[:, 76:88] = 0  # Pose
                
        if random.random() < 0.05:
            # Drop all hand landmarks
            landmarks[:, 88:] = 0
            
        return landmarks
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load and normalize landmarks
        landmarks = self.data[row['sequence_id']]
        landmarks = self.normalize_landmarks(landmarks)
        
        if self.augment:
            landmarks = self.augment_landmarks(landmarks)
            
        # Pad or resize sequence to max_len
        T = landmarks.shape[0]
        if T > self.max_len:
            # Resize if too long
            landmarks = F.interpolate(
                landmarks.permute(2, 0, 1)[None],
                size=self.max_len,
                mode='linear',
                align_corners=False
            )[0].permute(1, 2, 0)
        else:
            # Pad if too short
            pad_len = self.max_len - T
            landmarks = F.pad(landmarks, (0, 0, 0, 0, 0, pad_len))
            
        # Tokenize phrase
        tokens = self.tokenizer.encode(row['phrase'])
        
        return {
            'landmarks': landmarks,
            'tokens': tokens,
            'phrase': row['phrase'],
            'length': torch.tensor(T)
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    # Pad sequences to max length in batch
    max_token_len = max(item['tokens'].size(0) for item in batch)
    
    # Prepare tensors
    landmarks = torch.stack([item['landmarks'] for item in batch])
    tokens = torch.stack([
        F.pad(item['tokens'], (0, max_token_len - item['tokens'].size(0)), value=0)
        for item in batch
    ])
    lengths = torch.stack([item['length'] for item in batch])
    
    return {
        'landmarks': landmarks,
        'tokens': tokens,
        'phrase': [item['phrase'] for item in batch],
        'length': lengths
    }

# Model components from previous artifact: FeatureExtractor, RotaryPositionalEmbedding, 
# MultiHeadAttention, ConformerBlock, SqueezeformerBlock, ASLTranslationModel...

class ASLTranslationLoss(nn.Module):
    """Combined loss function for sequence prediction and confidence score"""
    def __init__(self, pad_idx: int = 0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: torch.Tensor,
        confidence_target: torch.Tensor
    ) -> torch.Tensor:
        # Main sequence prediction loss
        seq_loss = self.criterion(
            pred.view(-1, pred.size(-1)),
            target.view(-1)
        )
        
        # Confidence prediction loss (MSE)
        conf_loss = F.mse_loss(confidence, confidence_target)
        
        # Combine losses
        return seq_loss + 0.1 * conf_loss