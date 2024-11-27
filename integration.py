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
    
class Trainer:
    """Training controller for ASL Translation model"""
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
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        
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
        
        # Best validation score
        self.best_score = float('-inf')
        
    def get_scheduler(self):
        """Create cosine learning rate scheduler with warmup"""
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=self.num_training_steps,
            pct_start=self.num_warmup_steps / self.num_training_steps,
            anneal_strategy='cos',
            cycle_momentum=False
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            self.optimizer.zero_grad()
            
            # Move batch to device
            landmarks = batch['landmarks'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            lengths = batch['length'].to(self.device)
            
            # Create mask based on sequence lengths
            mask = torch.arange(landmarks.size(1))[None, :] < lengths[:, None]
            mask = mask.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                pred, confidence = self.model(landmarks, mask, tokens[:, :-1])
                
                # Calculate normalized Levenshtein distance for confidence target
                with torch.no_grad():
                    confidence_target = torch.tensor([
                        1 - Levenshtein.distance(
                            self.tokenizer.decode(p.argmax(-1)),
                            true_text
                        ) / max(len(true_text), 1)
                        for p, true_text in zip(pred, batch['phrase'])
                    ]).to(self.device)
                
                # Calculate loss
                loss = self.criterion(pred, tokens[:, 1:], confidence, confidence_target)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Clip gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer and scheduler step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model and compute metrics"""
        self.model.eval()
        total_loss = 0
        predictions = []
        ground_truth = []
        
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
        
        return avg_loss, avg_score
    
    def train(self, save_dir: str):
        """Train model for specified number of epochs"""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.num_training_steps // len(self.train_loader)):
            print(f"\nEpoch {epoch + 1}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_score = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Score: {val_score:.4f}")
            
            # Save best model
            if val_score > self.best_score:
                self.best_score = val_score
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_score': self.best_score,
                    },
                    os.path.join(save_dir, 'best_model.pt')
                )
                print(f"Saved new best model with score: {val_score:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_score': self.best_score,
                    },
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
                )

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    config = {
        'data_dir': '/kaggle/input/asl-fingerspelling/train_landmarks',  # Directory containing parquet files
        'metadata_path': '/kaggle/input/asl-fingerspelling/train.csv',
        'vocab_path': '/kaggle/input/asl-fingerspelling/character_to_prediction_index.json',
        'save_dir': '/kaggle/working/models',
        'batch_size': 32,
        'max_len': 384,
        'num_workers': 2,
        'learning_rate': 0.0045,
        'weight_decay': 0.08,
        'warmup_epochs': 10,
        'max_epochs': 400,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'wandb_project': 'asl-translation',
        'wandb_entity': None,
        'run_name': f'asl-translation-{time.strftime("%Y%m%d-%H%M%S")}',
    }

    # Make sure save directory exists
    os.makedirs(config['save_dir'], exist_ok=True)
        
    # Initialize tokenizer
    tokenizer = ASLTokenizer(config['vocab_path'])
    
    # Create datasets
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
    
    # Create dataloaders
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
    model = ASLTranslationModel(
        num_landmarks=130,
        feature_dim=208,
        num_classes=len(tokenizer.char_to_idx),
        num_layers=7,
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs'],
        max_epochs=config['max_epochs'],
        device=config['device']
    )
    
    # Train model
    trainer.train(config['save_dir'])

if __name__ == "__main__":
    main()