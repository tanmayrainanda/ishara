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
            
            