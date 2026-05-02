import subprocess
import time
import math

class ADBTouchController:
    def __init__(self, device_id=None):
        """
        Initialize ADB touch controller
        
        Args:
            device_id: Optional device ID for multiple devices
        """
        self.device_arg = ['-s', device_id] if device_id else []
        self.check_adb()
    
    def check_adb(self):
        """Check if ADB is available"""
        try:
            subprocess.run(['adb', 'version'], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            raise Exception("ADB not found. Please install ADB and add to PATH")
    
    def run_adb_command(self, command):
        """Execute ADB command"""
        cmd = ['adb'] + self.device_arg + command
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    
    def tap(self, x, y):
        """Single tap at coordinates"""
        self.run_adb_command(['shell', 'input', 'tap', str(x), str(y)])
    
    def swipe(self, x1, y1, x2, y2, duration_ms=300):
        """
        Swipe from (x1, y1) to (x2, y2)
        
        Args:
            duration_ms: Duration of swipe in milliseconds
        """
        self.run_adb_command([
            'shell', 'input', 'swipe', 
            str(x1), str(y1), str(x2), str(y2), 
            str(duration_ms)
        ])
    
    def long_press(self, x, y, duration_ms=1000):
        """Long press at coordinates"""
        self.swipe(x, y, x, y, duration_ms)
    
    def drag_and_drop(self, x1, y1, x2, y2, duration_ms=500):
        """Drag from one point to another"""
        self.swipe(x1, y1, x2, y2, duration_ms)
    
    def multi_touch(self, points, duration_ms=100):
        """
        Multi-touch gesture
        
        Args:
            points: List of (x, y) coordinates
            duration_ms: Duration of touch
        """
        if len(points) > 10:
            raise ValueError("Maximum 10 touch points supported")
        
        # Start multi-touch
        cmd = ['shell', 'input', 'touchscreen', 'swipe']
        
        # Add all points
        for x, y in points:
            cmd.extend([str(x), str(y)])
        
        cmd.append(str(duration_ms))
        self.run_adb_command(cmd)
    
    def get_screen_resolution(self):
        """Get device screen resolution"""
        output = self.run_adb_command(['shell', 'wm', 'size'])
        resolution = output.split(': ')[1]
        width, height = map(int, resolution.split('x'))
        return width, height
    
    def get_screen_density(self):
        """Get screen density (DPI)"""
        output = self.run_adb_command(['shell', 'wm', 'density'])
        density = int(output.split(': ')[1])
        return density

# Example usage
if __name__ == "__main__":
    touch = ADBTouchController()
    
    # Get screen info
    width, height = touch.get_screen_resolution()
    print(f"Screen resolution: {width}x{height}")
    
    # Basic actions
    touch.tap(500, 500)  # Tap at center
    time.sleep(1)
    
    touch.swipe(100, 500, 900, 500, 300)  # Horizontal swipe
    time.sleep(1)
    
    touch.long_press(500, 500, 2000)  # 2 second long press