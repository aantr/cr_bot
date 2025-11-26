import socket
import cv2
import numpy as np
import subprocess
import struct
import threading
import time

class HighPerformanceMinicap:
    def __init__(self, quality=10, fps=60):
        self.quality = quality
        self.fps = fps
        self.socket = None
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_stream(self):
        """Start high-performance minicap stream"""
        portrait = True
       
        # Stop any existing minicap
        subprocess.run(['adb', 'shell', 'pkill', '-f', 'minicap'])
        
        # Get device resolution
        result = subprocess.run(['adb', 'shell', 'wm', 'size'], 
                              capture_output=True, text=True)
        resolution = result.stdout.strip().split(': ')[1]
        width, height = map(int, resolution.split('x'))
        if portrait:

            width, height = height, width
        # Start minicap with JPG encoding and high FPS
        width_ = 720
        height_ = 1080
        cmd = (
            f"LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/minicap "
            f"-P {width}x{height}@{width_}x{height_}/0 "
            f"-Q {self.quality} "
            f"-r {self.fps} "
            f"-S"
        )
        
        subprocess.Popen(['adb', 'shell', cmd])
        
        # Forward port
        subprocess.run(['adb', 'forward', 'tcp:1313', 'localabstract:minicap'])
        
        time.sleep(2)  # Wait for minicap to start
        
        # Connect to stream
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('localhost', 1313))
        self.socket.settimeout(5.0)
        self.running = True
        
        # Read and parse banner
        self._parse_banner()
        
    def _parse_banner(self):
        """Parse minicap banner for stream information"""
        banner = self.socket.recv(24)
        if len(banner) == 24:
            (
                version, size,
                pid, real_width, real_height,
                virtual_width, virtual_height,
                orientation, quirks
            ) = struct.unpack("<2B5I2B", banner)
            
            print(f"Minicap: {real_width}x{real_height} (virtual: {virtual_width}x{virtual_height})")
            print(f"Orientation: {orientation}, Version: {version}")
            
    def stream_frames(self):
        """Stream frames with high performance"""
        while self.running:
            try:
                # Read frame length (4 bytes)
                length_bytes = self.socket.recv(4)
                if len(length_bytes) != 4:
                    break
                    
                frame_length = struct.unpack("<I", length_bytes)[0]
                
                # Read frame data
                frame_data = b""
                while len(frame_data) < frame_length:
                    chunk = self.socket.recv(min(4096, frame_length - len(frame_data)))
                    if not chunk:
                        break
                    frame_data += chunk
                
                if len(frame_data) == frame_length:
                    # Decode JPEG to OpenCV frame
                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        self.frame_count += 1
                        
                        # Calculate FPS
                        elapsed = time.time() - self.start_time
                        if elapsed > 1:
                            fps = self.frame_count / elapsed
                            self.frame_count = 0
                            self.start_time = time.time()
                        yield frame
                        
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Stream error: {e}")
                break
                
    def stop(self):
        """Stop the stream"""
        self.running = False
        if self.socket:
            self.socket.close()
        subprocess.run(['adb', 'shell', 'pkill', '-f', 'minicap'])

# Usage example
def main():
    minicap = HighPerformanceMinicap(quality=90, fps=60)
    minicap.start_stream()
    
    try:
        for frame in minicap.stream_frames():
            cv2.imshow('High-Performance Minicap', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        minicap.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()