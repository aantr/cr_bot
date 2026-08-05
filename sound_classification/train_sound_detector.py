# train_sound_detector.py - Complete working version
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import pickle
from tqdm import tqdm
import warnings
import random
import sys
warnings.filterwarnings('ignore')

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("WARNING: CUDA not available, using CPU (training will be slow)")

# ============================================
# SAFE AUDIO LOADER
# ============================================
def load_audio_safe(file_path, target_sr=22050):
    """
    Safe audio loading function that works on Windows.
    Uses soundfile with fallbacks.
    """
    try:
        audio, sr = sf.read(file_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return torch.from_numpy(audio).float().unsqueeze(0), target_sr
    except Exception as e1:
        try:
            # Fallback: torchaudio with sox backend
            torchaudio.set_audio_backend("sox_io")
            waveform, sr = torchaudio.load(file_path)
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return waveform.float(), target_sr
        except Exception as e2:
            try:
                # Last resort: scipy
                from scipy.io import wavfile
                sr, audio = wavfile.read(file_path)
                audio = audio.astype(np.float32) / 32768.0
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                if sr != target_sr:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                return torch.from_numpy(audio).float().unsqueeze(0), target_sr
            except Exception as e3:
                raise RuntimeError(f"Failed to load {file_path}: {e3}")

# ============================================
# MODEL ARCHITECTURE
# ============================================
class SoundDetectorCNN(nn.Module):
    """Optimized CNN for 3-second sound classification"""
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Dropout2d(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================
# AUDIO AUGMENTATION (CPU-based)
# ============================================
class AudioAugmenter:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def time_shift(self, waveform, shift_limit=0.2):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        shift = int(np.random.uniform(-shift_limit, shift_limit) * waveform.shape[-1])
        return torch.from_numpy(np.roll(waveform, shift))
    
    def add_noise(self, waveform, noise_factor=0.01):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        noise = np.random.randn(*waveform.shape) * noise_factor
        return torch.from_numpy(waveform + noise)
    
    def time_stretch(self, waveform, rate_limit=0.2):
        try:
            import librosa
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            rate = 1.0 + np.random.uniform(-rate_limit, rate_limit)
            stretched = librosa.effects.time_stretch(waveform.squeeze(), rate=rate)
            if len(stretched) > waveform.shape[-1]:
                stretched = stretched[:waveform.shape[-1]]
            else:
                stretched = np.pad(stretched, (0, waveform.shape[-1] - len(stretched)))
            return torch.from_numpy(stretched).unsqueeze(0)
        except:
            return torch.from_numpy(waveform) if isinstance(waveform, np.ndarray) else waveform
    
    def pitch_shift(self, waveform, n_steps=2):
        try:
            import librosa
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.numpy()
            shifted = librosa.effects.pitch_shift(
                waveform.squeeze(), 
                sr=self.sample_rate, 
                n_steps=np.random.uniform(-n_steps, n_steps)
            )
            return torch.from_numpy(shifted).unsqueeze(0)
        except:
            return torch.from_numpy(waveform) if isinstance(waveform, np.ndarray) else waveform
    
    def apply_augmentations(self, waveform):
        if torch.is_tensor(waveform):
            aug_waveform = waveform.clone()
        else:
            aug_waveform = waveform.copy()
        
        if torch.is_tensor(aug_waveform):
            aug_waveform = aug_waveform.numpy()
        
        if np.random.random() < 0.7:
            aug_type = np.random.choice(['noise', 'shift', 'stretch', 'pitch'], 
                                       p=[0.3, 0.3, 0.2, 0.2])
            if aug_type == 'noise':
                aug_waveform = self.add_noise(aug_waveform)
            elif aug_type == 'shift':
                aug_waveform = self.time_shift(aug_waveform)
            elif aug_type == 'stretch':
                aug_waveform = self.time_stretch(aug_waveform)
            else:
                aug_waveform = self.pitch_shift(aug_waveform)
        
        if isinstance(aug_waveform, np.ndarray):
            aug_waveform = torch.from_numpy(aug_waveform).float()
        return torch.clamp(aug_waveform, -1.0, 1.0)

# ============================================
# DATASET (CPU tensors)
# ============================================
class SoundDataset(Dataset):
    def __init__(self, file_paths, labels, label_encoder, 
                 sample_rate=22050, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.label_encoder = label_encoder
        self.sample_rate = sample_rate
        self.augment = augment
        
        # Transforms kept on CPU
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        if augment:
            self.augmenter = AudioAugmenter(sample_rate)
        
        # Validate files exist
        valid_files = []
        valid_labels = []
        for fp, lbl in zip(file_paths, labels):
            if Path(fp).exists():
                valid_files.append(fp)
                valid_labels.append(lbl)
        self.file_paths = valid_files
        self.labels = valid_labels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            waveform, sr = load_audio_safe(self.file_paths[idx], self.sample_rate)
        except Exception as e:
            print(f"Error loading {self.file_paths[idx]}: {e}")
            # Return zeros
            mel = torch.zeros(1, 128, 130)  # typical mel size
            return mel, torch.tensor(0, dtype=torch.long)
        
        # Ensure exactly 3 seconds
        target_length = self.sample_rate * 3
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.shape[1] < target_length:
            padding = torch.zeros(1, target_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        
        if self.augment:
            waveform = self.augmenter.apply_augmentations(waveform)
        
        # Mel spectrogram on CPU
        mel_spec = self.mel_transform(waveform.float())
        mel_spec_db = self.amplitude_to_db(mel_spec)     # CPU
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        label = self.label_encoder.transform([self.labels[idx]])[0]
        return mel_spec_db, torch.tensor(label, dtype=torch.long)

# ============================================
# DATA PREPARATION
# ============================================
def generate_synthetic_dataset(data_dir, samples_per_class=30):
    data_path = Path(data_dir)
    classes = ['alarm', 'doorbell', 'glass_break', 'dog_bark', 'car_horn']
    sr = 22050
    duration = 3.0
    
    for class_name in classes:
        class_dir = data_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_class):
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            audio = np.zeros_like(t)
            
            if class_name == 'alarm':
                base_freq = 800 + np.random.uniform(-200, 200)
                for j in range(4):
                    start_time = j * 0.75
                    end_time = start_time + 0.6
                    mask = (t >= start_time) & (t < end_time)
                    freq_mod = 300 * np.sin(2 * np.pi * 6 * t[mask])
                    audio[mask] += np.sin(2 * np.pi * (base_freq + freq_mod) * t[mask]) * 0.35
                    
            elif class_name == 'doorbell':
                freq1 = 750 + np.random.uniform(-100, 100)
                freq2 = 1000 + np.random.uniform(-100, 100)
                envelope = np.exp(-t * 1.5) + 0.5 * np.exp(-t * 0.5)
                audio = (np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t)) * 0.3 * envelope
                
            elif class_name == 'glass_break':
                from scipy import signal
                noise = np.random.randn(len(t))
                sos = signal.butter(4, [2000, 6000], btype='band', fs=sr, output='sos')
                noise = signal.sosfilt(sos, noise)
                envelope = np.exp(-t * 12) + 0.2 * np.exp(-t * 3)
                audio = noise * envelope * 0.3
                
            elif class_name == 'dog_bark':
                for j in range(np.random.randint(2, 5)):
                    start = np.random.uniform(0, 2)
                    dur = np.random.uniform(0.3, 0.8)
                    mask = (t >= start) & (t < start + dur)
                    freq = 400 + np.random.uniform(-100, 300)
                    env = np.exp(-(t[mask] - start) * 6)
                    audio[mask] += np.sin(2 * np.pi * freq * (t[mask] - start)) * env * 0.6
                    
            else:  # car_horn
                base_freq = 350 + np.random.uniform(-50, 50)
                for h in [1, 2, 3, 4]:
                    amp = 0.3 / h
                    audio += np.sin(2 * np.pi * base_freq * h * t) * amp
                audio *= (1 + 0.1 * np.sin(2 * np.pi * 5 * t))
            
            audio += np.random.randn(len(t)) * 0.015
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.85
            
            # Fade in/out
            fade_len = int(0.01 * sr)
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)
            
            output_path = class_dir / f"{class_name}_{i+1:03d}.wav"
            sf.write(output_path, audio.astype(np.float32), sr)
    
    print(f"Created {samples_per_class * len(classes)} synthetic audio files")

def prepare_dataset(data_dir):
    data_path = Path(data_dir)
    
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"Creating synthetic dataset at {data_dir}")
        data_path.mkdir(parents=True, exist_ok=True)
        generate_synthetic_dataset(data_dir, samples_per_class=30)
    
    # Collect audio files
    file_paths = []
    labels = []
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            for ext in ['*.wav', '*.flac', '*.ogg']:
                for audio_file in class_dir.glob(ext):
                    file_paths.append(str(audio_file))
                    labels.append(class_dir.name)
    
    if len(file_paths) == 0:
        raise ValueError(f"No audio files found in {data_dir}")
    
    from collections import Counter
    class_counts = Counter(labels)
    print(f"Found {len(file_paths)} audio files in {len(set(labels))} classes")
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Fit label encoder on ALL labels before splitting
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    print(f"Encoded classes: {list(label_encoder.classes_)}")
    
    # Split with fallback
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            file_paths, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    except:
        print("Using random split (too few samples for stratification)")
        indices = list(range(len(file_paths)))
        random.shuffle(indices)
        n_train = int(0.7 * len(indices))
        n_val = int(0.15 * len(indices))
        X_train = [file_paths[i] for i in indices[:n_train]]
        y_train = [labels[i] for i in indices[:n_train]]
        X_val = [file_paths[i] for i in indices[n_train:n_train+n_val]]
        y_val = [labels[i] for i in indices[n_train:n_train+n_val]]
        X_test = [file_paths[i] for i in indices[n_train+n_val:]]
        y_test = [labels[i] for i in indices[n_train+n_val:]]
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder

# ============================================
# TRAINING FUNCTION
# ============================================
def train_model(data_dir='dataset', num_epochs=50, batch_size=16, learning_rate=0.001):
    print("\n" + "="*50)
    print("Preparing Dataset")
    print("="*50)
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = prepare_dataset(data_dir)
    
    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Create datasets (no device argument)
    train_dataset = SoundDataset(X_train, y_train, label_encoder, augment=True)
    val_dataset = SoundDataset(X_val, y_val, label_encoder, augment=False)
    test_dataset = SoundDataset(X_test, y_test, label_encoder, augment=False)
    
    actual_batch_size = min(batch_size, max(2, len(train_dataset) // 4))
    
    # DataLoaders with pin_memory only if CUDA
    train_loader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    model = SoundDetectorCNN(num_classes=num_classes).to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    
    print("\n" + "="*50)
    print(f"Starting Training (batch_size={actual_batch_size})")
    print("="*50)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*train_correct/train_total:.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss_avg:.4f}, Train Acc={train_acc:.1f}%, "
              f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.1f}%")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'label_encoder': label_encoder,
                'history': history
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Test
    print("\n" + "="*50)
    print("Testing Best Model")
    print("="*50)
    
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"\nTest Accuracy: {test_acc:.1f}%")
    
    # Export final model
    export_model = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'label_encoder': label_encoder,
        'config': {
            'sample_rate': 22050,
            'duration': 3.0
        }
    }
    torch.save(export_model, 'sound_detector_final.pth')
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining complete! Files saved:")
    print("  - best_model.pth")
    print("  - sound_detector_final.pth")
    print("  - label_encoder.pkl")
    print("  - training_history.json")
    return model, history

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SOUND DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print("="*60)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    config = {
        'data_dir': 'dataset',
        'num_epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.001
    }
    
    try:
        model, history = train_model(**config)
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()