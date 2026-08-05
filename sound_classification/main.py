import torch
import torchaudio
import numpy as np
import librosa
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class SoundProfile:
    """Pre-calculated sound profile for fast matching"""
    name: str
    mel_spectrogram: torch.Tensor
    mfcc: torch.Tensor
    spectral_centroid: float
    spectral_bandwidth: float
    zero_crossing_rate: float
    rms_energy: float
    onset_strength: torch.Tensor
    chroma: torch.Tensor
    duration: float
    
class SoundPrecalculator:
    def __init__(self, device='cuda', sample_rate=22050):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        ).to(self.device)
        
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}
        ).to(self.device)
        
    def precalculate_sound(self, audio_path: str, sound_name: str) -> SoundProfile:
        """Pre-calculate all features for a sound file"""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.to(self.device)
        
        # Calculate spectral features
        mel_spec = self.mel_transform(waveform)
        mfcc = self.mfcc_transform(waveform)
        
        # Time-domain features (CPU for librosa compatibility)
        waveform_cpu = waveform.cpu().numpy().squeeze()
        
        return SoundProfile(
            name=sound_name,
            mel_spectrogram=mel_spec,
            mfcc=mfcc,
            spectral_centroid=librosa.feature.spectral_centroid(y=waveform_cpu, sr=self.sample_rate).mean(),
            spectral_bandwidth=librosa.feature.spectral_bandwidth(y=waveform_cpu, sr=self.sample_rate).mean(),
            zero_crossing_rate=librosa.feature.zero_crossing_rate(waveform_cpu).mean(),
            rms_energy=librosa.feature.rms(y=waveform_cpu).mean(),
            onset_strength=torch.from_numpy(
                librosa.onset.onset_strength(y=waveform_cpu, sr=self.sample_rate)
            ).to(self.device),
            chroma=torch.from_numpy(
                librosa.feature.chroma_stft(y=waveform_cpu, sr=self.sample_rate)
            ).to(self.device),
            duration=len(waveform_cpu) / self.sample_rate
        )
    
    def save_profiles(self, profiles: List[SoundProfile], output_path: str):
        """Save pre-calculated profiles to disk"""
        # Move tensors to CPU for serialization
        serializable_profiles = []
        for profile in profiles:
            serializable_profiles.append({
                'name': profile.name,
                'mel_spectrogram': profile.mel_spectrogram.cpu().numpy(),
                'mfcc': profile.mfcc.cpu().numpy(),
                'spectral_centroid': profile.spectral_centroid,
                'spectral_bandwidth': profile.spectral_bandwidth,
                'zero_crossing_rate': profile.zero_crossing_rate,
                'rms_energy': profile.rms_energy,
                'onset_strength': profile.onset_strength.cpu().numpy(),
                'chroma': profile.chroma.cpu().numpy(),
                'duration': profile.duration
            })
        
        with open(output_path, 'wb') as f:
            pickle.dump(serializable_profiles, f)


import torch.nn as nn
import torch.nn.functional as F

class SoundDetectorCNN(nn.Module):
    """CNN-based sound detector optimized for RTX 5080"""
    def __init__(self, num_classes: int, input_channels=1):
        super().__init__()
        
        # Convolutional layers with batch norm for faster training
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class SiameseSoundMatcher(nn.Module):
    """Siamese network for similarity-based matching"""
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # Shared embedding network
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
        
    def forward_once(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return F.normalize(x, p=2, dim=1)
    
    def forward(self, anchor, positive, negative):
        anchor_embed = self.forward_once(anchor)
        positive_embed = self.forward_once(positive)
        negative_embed = self.forward_once(negative)
        return anchor_embed, positive_embed, negative_embed


import sounddevice as sd
import threading
import queue
from collections import deque
import time

class RealtimeDetector:
    def __init__(self, 
                 model_path: str,
                 profiles_path: str,
                 sample_rate=22050,
                 chunk_duration=3.0,  # Match your sound duration
                 overlap=0.5):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(self.chunk_size * overlap)
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Load pre-calculated profiles
        with open(profiles_path, 'rb') as f:
            self.profiles = pickle.load(f)
        
        # Convert profiles to tensors on GPU
        self.profile_tensors = []
        for profile in self.profiles:
            self.profile_tensors.append({
                'name': profile['name'],
                'mel': torch.from_numpy(profile['mel_spectrogram']).to(self.device),
                'mfcc': torch.from_numpy(profile['mfcc']).to(self.device),
                'embedding': None  # Will be computed
            })
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_size * 2)
        self.output_queue = queue.Queue()
        self.is_running = False
        
        # Performance metrics
        self.inference_times = deque(maxlen=100)
        
        print(f"Detector initialized on {self.device}")
        print(f"CUDA Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = SoundDetectorCNN(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for real-time audio input"""
        if status:
            print(f"Audio callback status: {status}")
        self.audio_buffer.extend(indata[:, 0])
    
    def process_audio_chunk(self):
        """Process buffered audio and detect sounds"""
        if len(self.audio_buffer) < self.chunk_size:
            return None
        
        # Get chunk
        audio_chunk = np.array(list(self.audio_buffer))[:self.chunk_size]
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        # Compute mel spectrogram
        start_time = time.time()
        
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        ).to(self.device)(audio_tensor)
        
        # Add batch dimension if needed
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
        
        # Run inference
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Use mixed precision for RTX 5080
                predictions = self.model(mel_spec)
                probabilities = F.softmax(predictions, dim=1)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Get top prediction
        conf, predicted = torch.max(probabilities, 1)
        
        if conf > 0.7:  # Confidence threshold
            return {
                'class_id': predicted.item(),
                'confidence': conf.item(),
                'timestamp': time.time(),
                'inference_time': inference_time
            }
        
        return None
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.is_running:
            result = self.process_audio_chunk()
            if result:
                self.output_queue.put(result)
            
            # Print performance stats
            if len(self.inference_times) > 0:
                avg_time = np.mean(self.inference_times) * 1000
                if len(self.inference_times) % 10 == 0:
                    print(f"Avg inference: {avg_time:.2f}ms | "
                          f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            
            time.sleep(0.01)  # Small sleep to prevent CPU overload
    
    def start(self):
        """Start real-time detection"""
        self.is_running = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=1024,
            device=None  # Use default input device
        )
        self.stream.start()
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.start()
        
        print("Real-time detection started")
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        self.stream.stop()
        self.stream.close()
        self.detection_thread.join()