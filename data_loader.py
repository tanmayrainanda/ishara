import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import random

class ASLDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        char_to_pred_path: str,
        max_frames: int = 384,
        subset: str = "train",
        fold: Optional[int] = None,
        augment: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        ASL Fingerspelling Dataset loader with feature extraction
        
        Args:
            data_dir: Directory containing landmark files
            metadata_path: Path to metadata CSV
            char_to_pred_path: Path to character mapping JSON
            max_frames: Maximum number of frames to use
            subset: 'train' or 'valid'
            fold: Fold number for cross-validation
            augment: Whether to apply augmentations
            cache_dir: Directory to cache processed landmarks
        """
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        if fold is not None:
            # Split by signer (participant_id)
            unique_signers = self.metadata['participant_id'].unique()
            n_signers = len(unique_signers)
            fold_size = n_signers // 4
            valid_signers = unique_signers[fold * fold_size:(fold + 1) * fold_size]
            if subset == "train":
                self.metadata = self.metadata[~self.metadata['participant_id'].isin(valid_signers)]
            else:
                self.metadata = self.metadata[self.metadata['participant_id'].isin(valid_signers)]
        
        # Load character mapping
        with open(char_to_pred_path, 'r') as f:
            self.char_to_pred = json.load(f)
        self.pred_to_char = {v: k for k, v in self.char_to_pred.items()}
        
        # Define landmark groups
        self.landmark_groups = {
            'face': list(range(468)),
            'left_hand': list(range(21)),
            'pose': list(range(33)),
            'right_hand': list(range(21))
        }
        
        # Select specific landmarks as per winning solution
        self.selected_landmarks = {
            'face': list(range(76)),  # lips, nose, eyes
            'left_hand': list(range(21)),
            'right_hand': list(range(21)),
            'pose': list(range(6))  # arms only
        }
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def _load_landmarks(self, idx: int) -> np.ndarray:
        """Load and preprocess landmarks for a single sequence"""
        row = self.metadata.iloc[idx]
        sequence_id = row['sequence_id']
        
        # Try loading from cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{sequence_id}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load from parquet file
        landmarks_file = self.data_dir / f"{row['path']}"
        df = pd.read_parquet(landmarks_file)
        sequence_data = df.loc[sequence_id]
        
        # Extract and reshape landmarks
        landmarks = []
        for group in ['face', 'left_hand', 'right_hand', 'pose']:
            group_landmarks = []
            for coord in ['x', 'y', 'z']:
                cols = [f"{coord}_{group}_{i}" for i in range(len(self.landmark_groups[group]))]
                group_data = sequence_data[cols].values
                group_landmarks.append(group_data)
            landmarks.append(np.stack(group_landmarks, axis=-1))
        
        # Combine all landmarks
        combined = np.concatenate([
            landmarks[0][:, :76],  # face (selected)
            landmarks[1],          # left_hand
            landmarks[2],          # right_hand
            landmarks[3][:, :6]    # pose (selected)
        ], axis=1)
        
        # Cache if directory provided
        if self.cache_dir:
            np.save(cache_path, combined)
            
        return combined
    
    def _apply_augmentations(self, landmarks: np.ndarray) -> np.ndarray:
        """Apply augmentations to landmark sequence"""
        if not self.augment:
            return landmarks
            
        # Time stretch (resize along time axis)
        if random.random() < 0.8:
            scale = random.uniform(0.8, 1.2)
            num_frames = int(landmarks.shape[0] * scale)
            indices = np.linspace(0, landmarks.shape[0] - 1, num_frames)
            landmarks = np.stack([landmarks[int(i)] for i in indices])
        
        # Random shift along time axis
        if random.random() < 0.5:
            shift = random.randint(-10, 10)
            if shift > 0:
                landmarks = np.pad(landmarks, ((0, shift), (0, 0), (0, 0)))[shift:, :, :]
            else:
                landmarks = np.pad(landmarks, ((-shift, 0), (0, 0), (0, 0)))[:shift, :, :]
        
        # Left-right flip
        if random.random() < 0.5:
            # Swap left and right hand landmarks
            left_start = 76  # after face landmarks
            right_start = left_start + 21
            temp = landmarks[:, left_start:left_start+21].copy()
            landmarks[:, left_start:left_start+21] = landmarks[:, right_start:right_start+21]
            landmarks[:, right_start:right_start+21] = temp
            # Flip x coordinates
            landmarks[:, :, 0] *= -1
            
        # Finger dropout
        if random.random() < 0.5:
            num_fingers = random.randint(2, 6)
            num_windows = random.randint(2, 3)
            for _ in range(num_windows):
                start_frame = random.randint(0, landmarks.shape[0] - 10)
                end_frame = start_frame + random.randint(5, 10)
                for _ in range(num_fingers):
                    finger_idx = random.randint(0, 20)
                    landmarks[start_frame:end_frame, 76+finger_idx] = 0  # Left hand
                    landmarks[start_frame:end_frame, 97+finger_idx] = 0  # Right hand
                    
        return landmarks
    
    def _encode_phrase(self, phrase: str) -> List[int]:
        """Convert phrase to token indices"""
        return [self.char_to_pred[c] for c in phrase]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int]]:
        row = self.metadata.iloc[idx]
        
        # Load and preprocess landmarks
        landmarks = self._load_landmarks(idx)
        
        # Apply augmentations
        landmarks = self._apply_augmentations(landmarks)
        
        # Pad or resize to max_frames
        num_frames = landmarks.shape[0]
        if num_frames > self.max_frames:
            # Resize using linear interpolation
            indices = np.linspace(0, num_frames - 1, self.max_frames)
            landmarks = np.stack([landmarks[int(i)] for i in indices])
        else:
            # Pad with zeros
            padding = np.zeros((self.max_frames - num_frames, landmarks.shape[1], landmarks.shape[2]))
            landmarks = np.concatenate([landmarks, padding], axis=0)
            
        # Normalize
        landmarks = (landmarks - landmarks.mean(axis=(0, 1), keepdims=True)) / (landmarks.std(axis=(0, 1), keepdims=True) + 1e-8)
        
        # Convert to tensor
        landmarks = torch.from_numpy(landmarks).float()
        
        # Encode phrase
        phrase_encoded = self._encode_phrase(row['phrase'])
        
        return landmarks, phrase_encoded

def get_dataloader(
    data_dir: str,
    metadata_path: str,
    char_to_pred_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """Create DataLoader for ASL dataset"""
    dataset = ASLDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        char_to_pred_path=char_to_pred_path,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=dataset_kwargs.get('subset', 'train') == 'train',
        pin_memory=True
    )