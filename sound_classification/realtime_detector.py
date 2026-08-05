# realtime_detector.py
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import pickle
from pathlib import Path
import sounddevice as sd
import queue
import threading
import time
from collections import deque

class RealtimeSoundDetector:
    def __init__(self, model_path='sound_detector_final.pth', 
                 label_encoder_path='label_encoder.pkl',
                 sample_rate=22050,
                 chunk_duration=3.0,
                 confidence_threshold=0.7):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.confidence_threshold = confidence_threshold
        
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        from train_sound_detector import SoundDetectorCNN
        self.model = SoundDetectorCNN(num_classes=checkpoint['num_classes'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        print("Loading label encoder...")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.classes = self.label_encoder.classes_
        print(f"Classes: {list(self.classes)}")
        
        # Audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB().to(self.device)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_size * 3)
        self.detection_queue = queue.Queue()
        self.is_running = False
        
        # Statistics
        self.detection_count = 0
        self.last_detection_time = 0
        self.cooldown_period = 1.0  # seconds between detections
        
        print("Ready for real-time detection!")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for real-time audio input"""
        if status:
            print(f"Status: {status}")
        self.audio_buffer.extend(indata[:, 0])
    
    def process_audio(self):
        """Process audio buffer and detect sounds"""
        while self.is_running:
            if len(self.audio_buffer) >= self.chunk_size:
                # Get audio chunk
                audio_chunk = np.array(list(self.audio_buffer))[-self.chunk_size:]
                
                # Convert to tensor
                waveform = torch.from_numpy(audio_chunk).float().unsqueeze(0)
                waveform = waveform.to(self.device)
                
                # Normalize
                if waveform.abs().max() > 0:
                    waveform = waveform / waveform.abs().max()
                
                # Compute mel spectrogram
                mel_spec = self.mel_transform(waveform)
                mel_spec_db = self.amplitude_to_db(mel_spec)
                mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                
                # Add batch dimension if needed
                if mel_spec_db.dim() == 3:
                    mel_spec_db = mel_spec_db.unsqueeze(1)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(mel_spec_db)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, dim=1)
                
                confidence_val = confidence.item()
                predicted_class = predicted.item()
                
                # Check confidence threshold and cooldown
                current_time = time.time()
                if (confidence_val >= self.confidence_threshold and 
                    current_time - self.last_detection_time > self.cooldown_period):
                    
                    class_name = self.classes[predicted_class]
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence_val,
                        'timestamp': current_time,
                        'all_probabilities': {
                            cls: prob.item() 
                            for cls, prob in zip(self.classes, probabilities[0])
                        }
                    }
                    
                    self.detection_queue.put(detection)
                    self.detection_count += 1
                    self.last_detection_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU overload
    
    def start(self, input_device=None):
        """Start real-time detection"""
        print("\nStarting real-time detection...")
        print("Press Ctrl+C to stop")
        
        self.is_running = True
        
        # List available devices if needed
        if input_device is None:
            print("\nAvailable audio devices:")
            print(sd.query_devices())
            input_device = sd.default.device[0]  # Use default input
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=input_device
            )
            self.stream.start()
            print("Audio stream started")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            print("Available devices:")
            print(sd.query_devices())
            return
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()
        print("Processing thread started")
        
        # Main loop - print detections
        try:
            while self.is_running:
                if not self.detection_queue.empty():
                    detection = self.detection_queue.get()
                    print(f"\n{'='*50}")
                    print(f"🔊 DETECTED: {detection['class'].upper()}")
                    print(f"   Confidence: {detection['confidence']:.2%}")
                    print(f"   Time: {time.strftime('%H:%M:%S', time.localtime(detection['timestamp']))}")
                    print(f"   All probabilities:")
                    for cls, prob in detection['all_probabilities'].items():
                        bar = '█' * int(prob * 20)
                        print(f"     {cls:15s}: {bar} {prob:.2%}")
                    print(f"   Total detections: {self.detection_count}")
                    print(f"{'='*50}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=2.0)
        print("Detection stopped")

# ============================================
# 2. FILE-BASED TESTING
# ============================================
def test_on_file(audio_path, model_path='sound_detector_final.pth', 
                 label_encoder_path='label_encoder.pkl'):
    """Test the model on a single audio file"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    from train_sound_detector import SoundDetectorCNN
    model = SoundDetectorCNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if sr != 22050:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    
    # Ensure 3 seconds
    target_length = 22050 * 3
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    
    # Convert to tensor
    waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    # Normalize
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()
    
    # Mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128
    ).to(device)
    amp_to_db = T.AmplitudeToDB().to(device)
    
    mel_spec = mel_transform(waveform)
    mel_spec_db = amp_to_db(mel_spec)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    if mel_spec_db.dim() == 3:
        mel_spec_db = mel_spec_db.unsqueeze(1)
    
    # Predict
    with torch.no_grad():
        outputs = model(mel_spec_db)
        probabilities = F.softmax(outputs, dim=1)
    
    # Show results
    print(f"\nResults for: {audio_path}")
    print("-" * 50)
    for i, (cls, prob) in enumerate(zip(label_encoder.classes_, probabilities[0])):
        bar = '█' * int(prob.item() * 20)
        print(f"{cls:15s}: {bar} {prob.item():.2%}")
    
    predicted = torch.argmax(probabilities).item()
    confidence = probabilities[0][predicted].item()
    print(f"\nPredicted: {label_encoder.classes_[predicted]} (confidence: {confidence:.2%})")

# ============================================
# 3. BATCH TESTING
# ============================================
def batch_test(test_dir, model_path='sound_detector_final.pth',
               label_encoder_path='label_encoder.pkl'):
    """Test model on a directory of audio files"""
    
    test_path = Path(test_dir)
    audio_files = list(test_path.glob('*.wav')) + list(test_path.glob('*.flac'))
    
    if not audio_files:
        print(f"No audio files found in {test_dir}")
        return
    
    print(f"Testing {len(audio_files)} files...")
    
    correct = 0
    total = 0
    
    for audio_file in audio_files:
        # Get true label from parent directory name if organized in folders
        true_label = audio_file.parent.name if audio_file.parent != test_path else None
        
        # Test file
        print(f"\nTesting: {audio_file.name}")
        test_on_file(str(audio_file), model_path, label_encoder_path)
        
        if true_label:
            total += 1
            # You would need to compare with prediction here
    
    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("REAL-TIME SOUND DETECTION SYSTEM")
    print("="*60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Test on a file
            if len(sys.argv) > 2:
                test_on_file(sys.argv[2])
            else:
                print("Usage: python realtime_detector.py test <audio_file>")
        elif sys.argv[1] == 'batch':
            # Batch test on directory
            if len(sys.argv) > 2:
                batch_test(sys.argv[2])
            else:
                batch_test('test_audio')
        else:
            print("Unknown command. Use 'test' or 'batch'")
    else:
        # Start real-time detection
        detector = RealtimeSoundDetector()
        detector.start()