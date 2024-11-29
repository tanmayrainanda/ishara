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
import time
import pyarrow.parquet as pq
import wandb
import sys

wandb.login(key="afe8b8c0a3f1c1339a3daa9f619cb7c311218022")
wandb.init(project="asl-translation")

class FeatureExtractor(nn.Module):
    def __init__(self, input_channels: int = 3, output_dim: int = 52):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, output_dim)

    def forward(self, x):
        # x shape: [batch, time, landmarks, channels]
        B, T, L, C = x.shape
        
        # Reshape for 1D convolution over landmarks
        x = x.permute(0, 1, 3, 2)  # [batch, time, channels, landmarks]
        x = x.reshape(B * T, C, L)  # [batch*time, channels, landmarks]
        
        # Apply convolution
        x = self.conv(x)  # [batch*time, 64, landmarks]
        x = self.bn(x)
        x = F.relu(x)
        
        # Global average pooling over landmarks
        x = x.mean(dim=2)  # [batch*time, 64]
        
        # Project to output dimension
        x = self.linear(x)  # [batch*time, output_dim]
        
        # Reshape back to sequence
        x = x.reshape(B, T, -1)  # [batch, time, output_dim]
        
        return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 384):
        super().__init__()
        head_dim = dim // 8  # Assuming 8 heads
        half_head_dim = head_dim // 2
        emb = math.log(10000) / (half_head_dim - 1)
        emb = torch.exp(torch.arange(half_head_dim) * -emb)
        pos = torch.arange(max_seq_len)
        emb = pos[:, None] * emb[None, :]  # [max_seq_len, half_head_dim]
        self.register_buffer('sin', emb.sin())
        self.register_buffer('cos', emb.cos())

    def forward(self, x):
        seq_len = x.shape[1]
        return self.sin[:seq_len], self.cos[:seq_len]

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def apply_rotary_pos_emb(self, q, k, sin, cos):
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        
        # Separate half of head dim for rotation
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        
        # Apply rotation using complementary pairs
        q = torch.cat([
            q1 * cos - q2 * sin,
            q2 * cos + q1 * sin,
        ], dim=-1)
        
        k = torch.cat([
            k1 * cos - k2 * sin,
            k2 * cos + k1 * sin,
        ], dim=-1)
        
        return q, k

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project inputs
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        q, k = self.apply_rotary_pos_emb(q, k, sin, cos)
        
        # Reshape for attention
        q = q.transpose(1, 2)  # [B, H, L, D/H]
        k = k.transpose(1, 2)  # [B, H, L, D/H]
        v = v.transpose(1, 2)  # [B, H, L, D/H]
        
        # Calculate attention scores
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            # Expand mask for attention heads
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine with values
        out = torch.matmul(attn, v)  # [B, H, L, D/H]
        out = out.transpose(1, 2).contiguous()  # [B, L, H, D/H]
        out = out.reshape(B, L, D)  # [B, L, D]
        
        return self.out_proj(out)

class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadAttention(dim, num_heads, dropout)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim*2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv = nn.Conv1d(dim, dim, 1)
        self.conv_dropout = nn.Dropout(dropout)
        
        # Feed forward module
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        
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
        x = self.conv_norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = residual + x * self.scale
        
        # Feed forward
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + x * self.scale
        
        return x

class SqueezeformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadAttention(dim, num_heads, dropout)
        
        # Feed forward modules
        self.ff1_norm = nn.LayerNorm(dim)
        self.ff1 = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim*2, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = nn.SiLU()
        self.pointwise_conv = nn.Conv1d(dim, dim, 1)
        self.conv_dropout = nn.Dropout(dropout)
        
        # Feed forward module 2
        self.ff2_norm = nn.LayerNorm(dim)
        self.ff2 = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First feed forward
        residual = x
        x = self.ff1_norm(x)
        x = self.ff1(x)
        x = residual + x * self.scale
        
        # Self attention
        residual = x
        x = self.norm1(x)
        x = self.mhsa(x, sin, cos, mask)
        x = self.dropout(x)
        x = residual + x * self.scale
        
        # Convolution module
        residual = x
        x = self.conv_norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = residual + x * self.scale
        
        # Second feed forward
        residual = x
        x = self.ff2_norm(x)
        x = self.ff2(x)
        x = residual + x * self.scale
        
        return x

class ASLTranslationModel(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 130,
        feature_dim: int = 208,
        num_classes: int = 59,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature extractors for different landmark types
        self.face_extractor = FeatureExtractor(3, 52)
        self.pose_extractor = FeatureExtractor(3, 52)
        self.left_hand_extractor = FeatureExtractor(3, 52)
        self.right_hand_extractor = FeatureExtractor(3, 52)
        
        # Target embedding
        self.target_embedding = nn.Embedding(num_classes, feature_dim)
        self.pos_embedding = RotaryPositionalEmbedding(feature_dim)
        
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
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Output layers
        self.confidence_head = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, L, C = x.shape
        
        # Extract features
        face = x[:, :, :76]
        pose = x[:, :, 76:88]
        left_hand = x[:, :, 88:109]
        right_hand = x[:, :, 109:]
        
        face_feats = self.face_extractor(face)
        pose_feats = self.pose_extractor(pose)
        left_hand_feats = self.left_hand_extractor(left_hand)
        right_hand_feats = self.right_hand_extractor(right_hand)
        
        features = torch.cat(
            [face_feats, pose_feats, left_hand_feats, right_hand_feats],
            dim=-1
        )
        
        # Process features through encoder
        sin, cos = self.pos_embedding(features)
        
        # (Continuing the ASLTranslationModel forward method)
        
        conformer_out = features
        squeezeformer_out = features
        
        # Convert mask to padding mask for encoder if provided
        encoder_padding_mask = None
        if mask is not None:
            encoder_padding_mask = mask  # [B, T]
        
        for conf_layer, squeeze_layer in zip(
            self.conformer_layers,
            self.squeezeformer_layers
        ):
            conformer_out = conf_layer(conformer_out, sin, cos, encoder_padding_mask)
            squeezeformer_out = squeeze_layer(squeezeformer_out, sin, cos, encoder_padding_mask)
        
        encoder_out = conformer_out + squeezeformer_out
        confidence = self.confidence_head(encoder_out[:, 0]).squeeze(-1)
        
        if tgt is not None:
            # Process target sequence
            tgt_embedded = self.target_embedding(tgt)  # [B, tgt_len, dim]
            tgt_embedded = self.dropout(tgt_embedded)
            
            # Create causal mask for target self-attention
            tgt_len = tgt.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(x.device)
            
            # Create memory padding mask for encoder-decoder attention
            memory_padding_mask = None
            if encoder_padding_mask is not None:
                memory_padding_mask = ~encoder_padding_mask  # Convert to padding mask format
            
            # Decoder forward pass
            decoder_out = self.decoder(
                tgt_embedded,
                encoder_out,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_padding_mask
            )
            output = self.classifier(decoder_out)
        else:
            output = self.classifier(encoder_out)
        
        return output, confidence

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate causal mask for decoder self-attention"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask

def train_step(
    model: ASLTranslationModel,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda',
    grad_scaler = None
) -> float:
    """Perform one training step"""
    optimizer.zero_grad()
    
    # Move batch to device
    landmarks = batch['landmarks'].to(device)
    tokens = batch['tokens'].to(device)
    mask = batch['mask'].to(device) if 'mask' in batch else None
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        pred, confidence = model(landmarks, mask, tokens[:, :-1])
        
        # Calculate confidence target (normalized Levenshtein distance)
        with torch.no_grad():
            confidence_target = torch.tensor([
                1 - Levenshtein.distance(
                    model.tokenizer.decode(p.argmax(-1).cpu()),
                    true_text
                ) / max(len(true_text), 1)
                for p, true_text in zip(pred, batch['phrase'])
            ]).to(device)
        
        # Calculate loss
        loss = criterion(pred, tokens[:, 1:], confidence, confidence_target)
    
    # Backward pass with gradient scaling
    if grad_scaler is not None:
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    return loss.item()

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
            pred.reshape(-1, pred.size(-1)),
            target.reshape(-1)
        )
        
        # Confidence prediction loss (MSE)
        conf_loss = F.mse_loss(confidence, confidence_target)
        
        # Combine losses
        return seq_loss + 0.1 * conf_loss

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
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

import pandas as pd
import pyarrow.parquet as pq

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
        
        # Get parquet files in directory
        self.parquet_files = sorted(list(Path(data_dir).glob('*.parquet')))
        print(f"Found {len(self.parquet_files)} parquet files")
        
        # Create mapping of sequence_id to parquet file
        self.sequence_to_file = {}
        for parquet_file in tqdm(self.parquet_files, desc='Indexing parquet files'):
            # Read sequence_ids from parquet file without loading data
            table = pq.read_table(parquet_file, columns=['sequence_id'])
            sequences = table['sequence_id'].to_pylist()
            for seq_id in sequences:
                self.sequence_to_file[seq_id] = parquet_file
                
        # Filter df to only include sequences we have data for
        self.df = self.df[self.df['sequence_id'].isin(self.sequence_to_file.keys())]
        print(f"Dataset contains {len(self.df)} sequences")
        
    def __len__(self) -> int:
        return len(self.df)

    def get_landmarks(self, sequence_id: str) -> np.ndarray:
        """Load landmarks for a specific sequence from parquet file"""
        parquet_file = self.sequence_to_file[sequence_id]
        
        # Read the specific sequence from parquet file
        table = pq.read_table(
            parquet_file,
            filters=[('sequence_id', '=', sequence_id)]
        )
        df = table.to_pandas()
        
        # Extract landmark columns (excluding sequence_id and frame)
        landmark_cols = [col for col in df.columns if col not in ['sequence_id', 'frame']]
        landmarks = df[landmark_cols].values
        
        # Reshape landmarks to (frames, landmarks, 3)
        num_landmarks = len(landmark_cols) // 3
        landmarks = landmarks.reshape(-1, num_landmarks, 3)
        
        return landmarks
        
    def augment_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to landmark sequence with safety checks"""
        T = landmarks.shape[0]  # sequence length
        
        # Time augmentations
        if random.random() < 0.8:
            # Random resize along time axis
            scale = random.uniform(0.8, 1.2)
            new_T = int(T * scale)
            if new_T > 0:  # Only resize if new length is valid
                indices = torch.linspace(0, T-1, new_T).long()
                landmarks = landmarks[indices]
                T = new_T  # Update sequence length
        
        if random.random() < 0.5 and T > 1:
            # Random time shift
            shift = random.randint(-min(5, T//2), min(5, T//2))
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
            ]).float()[None]  # Add batch dimension
            
            # Handle each frame separately
            landmarks_2d = landmarks[..., :2]  # Only x,y coordinates
            
            # Process each frame
            transformed_frames = []
            for t in range(T):
                frame = landmarks_2d[t:t+1]  # Add batch dimension
                
                grid = F.affine_grid(
                    theta,
                    size=(1, 1, frame.shape[1], 2),
                    align_corners=False
                )
                
                transformed = F.grid_sample(
                    frame[:, None],  # Add channel dimension
                    grid,
                    align_corners=False
                )
                transformed_frames.append(transformed[:, 0])  # Remove channel dimension
            
            # Stack frames back together
            landmarks_2d = torch.cat(transformed_frames, dim=0)
            landmarks[..., :2] = landmarks_2d
        
        # Landmark dropping
        if random.random() < 0.5 and T >= 20:  # Only apply to sequences long enough
            # Randomly drop fingers
            num_fingers = random.randint(1, 3)  # Reduced from 2-6 to 1-3
            num_windows = random.randint(1, 2)  # Reduced from 2-3 to 1-2
            
            for _ in range(num_windows):
                if T <= 20:  # Skip if sequence too short
                    break
                
                window_size = min(random.randint(5, 10), T-1)  # Reduced window size
                window_start = random.randint(0, T - window_size)
                window_end = window_start + window_size
                
                for _ in range(num_fingers):
                    finger_start = random.randint(88, 126)  # Adjusted range
                    landmarks[window_start:window_end, finger_start:finger_start+4] = 0
        
        if random.random() < 0.3:  # Reduced probability from 0.5
            # Drop face or pose landmarks
            if random.random() < 0.5:
                landmarks[:, :76] = 0  # Face
            else:
                landmarks[:, 76:88] = 0  # Pose
        
        if random.random() < 0.05:  # Keep rare complete hand dropping
            # Drop all hand landmarks
            landmarks[:, 88:] = 0
        
        return landmarks
    
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load landmarks
        landmarks = self.get_landmarks(row['sequence_id'])
        landmarks = self.normalize_landmarks(landmarks)
        
        if self.augment:
            landmarks = self.augment_landmarks(landmarks)
            
        # Pad or resize sequence to max_len
        T = landmarks.shape[0]
        if T > self.max_len:
            # Resize if too long using linear interpolation
            indices = torch.linspace(0, T-1, self.max_len).long()
            landmarks = landmarks[indices]
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
    
class Trainer:
    """Training controller for ASL Translation model with WandB logging"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: ASLTokenizer,
        learning_rate: float = 0.0045,
        weight_decay: float = 0.08,
        warmup_epochs: int = 10,
        max_epochs: int = 400,
        device: str = 'cuda',
        wandb_config: dict = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.max_epochs = max_epochs
        self.wandb_config = wandb_config
        
        # Initialize WandB
        # Initialize WandB
        if wandb_config:
            wandb.init(
                project=wandb_config['project'],
                name=wandb_config['run_name'],
                config={
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'warmup_epochs': warmup_epochs,
                    'max_epochs': max_epochs,
                    'batch_size': train_loader.batch_size,
                    'architecture': 'Conformer-Squeezeformer'
                }
            )
            # Watch model gradients
            wandb.watch(model, log_freq=100)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.num_training_steps = len(train_loader) * max_epochs
        self.num_warmup_steps = len(train_loader) * warmup_epochs
        self.scheduler = self.get_scheduler()
        
        # Loss function
        self.criterion = ASLTranslationLoss()
        
        # Gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
    def get_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=self.num_training_steps,
            pct_start=self.num_warmup_steps / self.num_training_steps,
            anneal_strategy='cos',
            cycle_momentum=False
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch with proper progress tracking and WandB logging"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc='Training',
            leave=True,
            position=0,
            dynamic_ncols=True
        )
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    self.optimizer.zero_grad()
                    
                    # Move batch to device
                    landmarks = batch['landmarks'].to(self.device)
                    tokens = batch['tokens'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    
                    # Create mask based on sequence lengths
                    seq_length = landmarks.size(1)
                    position_indices = torch.arange(seq_length, device=self.device)[None, :]
                    mask = position_indices < lengths[:, None]
                    
                    try:
                        with autocast():
                            pred, confidence = self.model(landmarks, mask, tokens[:, :-1])
                            
                            # Calculate confidence target
                            with torch.no_grad():
                                confidence_target = torch.tensor([
                                    1 - Levenshtein.distance(
                                        self.tokenizer.decode(p.argmax(-1).cpu()),
                                        true_text
                                    ) / max(len(true_text), 1)
                                    for p, true_text in zip(pred, batch['phrase'])
                                ], device=self.device)
                            
                            loss = self.criterion(pred, tokens[:, 1:], confidence, confidence_target)
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        
                        # Update metrics
                        total_loss += loss.item()
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        # Log to WandB
                        if self.wandb_config and batch_idx % 10 == 0:  # Log every 10 batches
                            wandb.log({
                                'batch_loss': loss.item(),
                                'learning_rate': current_lr,
                                'batch_confidence': confidence.mean().item(),
                                'batch_confidence_target': confidence_target.mean().item()
                            })
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'batch_loss': f'{loss.item():.4f}',
                            'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                            'lr': f'{current_lr:.2e}',
                            'batch': f'{batch_idx+1}/{num_batches}'
                        })
                        
                        if batch_idx % 10 == 0:
                            progress_bar.refresh()
                    
                    except RuntimeError as e:
                        print(f"\nError in forward/backward pass: {str(e)}")
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        continue
                    
                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {str(e)}")
                    continue
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            return total_loss / (batch_idx + 1) if batch_idx > 0 else float('inf')
        
        finally:
            progress_bar.close()
        
        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model and compute metrics"""
        self.model.eval()
        total_loss = 0
        predictions = []
        ground_truth = []
        confidence_scores = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # Move batch to device
            landmarks = batch['landmarks'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            lengths = batch['length'].to(self.device)
            
            # Create mask based on sequence lengths
            mask = torch.arange(landmarks.size(1))[None, :] < lengths[:, None]
            mask = mask.to(self.device)
            
            # Forward pass
            pred, confidence = self.model(landmarks, mask)
            confidence_scores.extend(confidence.cpu().tolist())
            
            # Decode predictions
            pred_texts = [
                self.tokenizer.decode(p.argmax(-1))
                for p in pred
            ]
            predictions.extend(pred_texts)
            ground_truth.extend(batch['phrase'])
            
            # Calculate loss
            confidence_target = torch.tensor([
                1 - Levenshtein.distance(pred_text, true_text) / max(len(true_text), 1)
                for pred_text, true_text in zip(pred_texts, batch['phrase'])
            ]).to(self.device)
            
            loss = self.criterion(pred, tokens[:, 1:], confidence, confidence_target)
            total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate normalized Levenshtein distance
        distances = [
            1 - Levenshtein.distance(pred, true) / max(len(pred), len(true))
            for pred, true in zip(predictions, ground_truth)
        ]
        avg_score = sum(distances) / len(distances)
        
        # Log validation examples to WandB
        if self.wandb_config:
            # Log validation metrics
            wandb.log({
                'val_loss': avg_loss,
                'val_score': avg_score,
                'val_confidence_mean': np.mean(confidence_scores),
                'val_confidence_std': np.std(confidence_scores)
            })
            
            # Log prediction examples
            examples = []
            for i in range(min(10, len(predictions))):
                examples.append(wandb.Table(
                    columns=['Ground Truth', 'Prediction', 'Score'],
                    data=[[ground_truth[i], predictions[i], distances[i]]]
                ))
            wandb.log({"prediction_examples": examples})
        
        return avg_loss, avg_score

    def train(self, save_dir: str):
        """Train model with improved logging and error handling"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nStarting training for {self.max_epochs} epochs")
        print(f"Training device: {self.device}")
        print(f"Number of training batches: {len(self.train_loader)}")
        print("=" * 50)
        
        best_val_score = float('-inf')
        
        try:
            for epoch in range(self.max_epochs):
                epoch_start_time = time.time()
                
                print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
                print("-" * 30)
                
                # Train epoch
                train_loss = self.train_epoch()
                epoch_time = time.time() - epoch_start_time
                
                # Print and log metrics
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"Training Loss: {train_loss:.4f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"Epoch Time: {epoch_time:.1f}s")
                
                if self.wandb_config:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'epoch_time': epoch_time
                    })
                
                # Validate and save checkpoint periodically
                if (epoch + 1) % 5 == 0:
                    print("\nRunning validation...")
                    val_loss, val_score = self.validate()
                    print(f"Validation Loss: {val_loss:.4f}")
                    print(f"Validation Score: {val_score:.4f}")
                    
                    # Save best model
                    if val_score > best_val_score:
                        best_val_score = val_score
                        checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'val_score': val_score,
                        }, checkpoint_path)
                        print(f"✓ Saved new best model (score: {val_score:.4f})")
                        
                        if self.wandb_config:
                            wandb.log({'best_val_score': val_score})
                
                # Save periodic checkpoint
                if (epoch + 1) % 40 == 0:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, checkpoint_path)
                    print(f"✓ Saved checkpoint at epoch {epoch+1}")
                
                # Force flush stdout
                sys.stdout.flush()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        except Exception as e:
            print(f"\nTraining failed with error: {str(e)}")
            raise
        
        finally:
            # Save final model
            final_path = os.path.join(save_dir, 'final_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_score': best_val_score,
            }, final_path)
            print(f"\n✓ Saved final model")
            
            # Finish WandB run
            if self.wandb_config:
                wandb.finish()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    
    
    # Configuration
    config = {
        'data_dir': '/kaggle/input/asl-fingerspelling/train_landmarks',
        'metadata_path': '/kaggle/input/asl-fingerspelling/train.csv',
        'vocab_path': '/kaggle/input/asl-fingerspelling/character_to_prediction_index.json',
        'save_dir': '/kaggle/working/models',
        'batch_size': 200,
        'max_len': 384,
        'num_workers': 2,
        'learning_rate': 0.0045,
        'weight_decay': 0.08,
        'warmup_epochs': 1,
        'max_epochs': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'wandb_project': 'asl-translation',
        'run_name': f'asl-translation-{time.strftime("%Y%m%d-%H%M%S")}'
    }

    # WandB configuration
    wandb_config = {
        'project': config['wandb_project'],
        'run_name': config['run_name']
    }

    print("\nInitializing training...")
    print(f"Device: {config['device']}")
    
    # Make sure save directory exists
    os.makedirs(config['save_dir'], exist_ok=True)
    
    try:
        # Initialize tokenizer
        print("Loading tokenizer...")
        tokenizer = ASLTokenizer(config['vocab_path'])
        
        # Create datasets
        print("\nCreating datasets...")
        train_dataset = ASLDataset(
            config['data_dir'],
            config['metadata_path'],
            tokenizer,
            max_len=config['max_len'],
            augment=True,
            mode='train'
        )
        
        val_dataset = ASLDataset(
            config['data_dir'],
            config['metadata_path'],
            tokenizer,
            max_len=config['max_len'],
            augment=False,
            mode='val'
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Create model
        print("\nInitializing model...")
        model = ASLTranslationModel(
            num_landmarks=130,
            feature_dim=208,
            num_classes=len(tokenizer.char_to_idx),
            num_layers=7,
            dropout=0.1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            max_epochs=config['max_epochs'],
            device=config['device'],
            wandb_config=wandb_config
        )
        
        # Train model
        print("\nStarting training...")
        trainer.train(config['save_dir'])

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise
    
    finally:
        # Clean up wandb
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()