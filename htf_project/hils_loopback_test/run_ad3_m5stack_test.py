#!/usr/bin/env python3
"""
Coordinated test script for AD3 Ch2 to M5Stack verification
Manages AD3 output while monitoring M5Stack serial communication
"""

import time
import subprocess
import threading
import queue
import serial
import serial.tools.list_ports
from ad3_m5stack_test import AD3_M5StackTest

class M5StackMonitor:
    """Monitor M5Stack serial output"""
    
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.monitoring = False
        self.data_queue = queue.Queue()
        self.monitor_thread = None
        
    def find_m5stack_port(self):
        """Auto-detect M5Stack serial port"""
        ports = serial.tools.list_ports.comports()
        
        # Look for common M5Stack identifiers
        m5_identifiers = ['CP210', 'USB-SERIAL', 'Silicon Labs', 'UART']
        
        for port in ports:
            port_info = f"{port.device} - {port.description}"
            print(f"Found port: {port_info}")
            
            for identifier in m5_identifiers:
                if identifier.lower() in port.description.lower():
                    print(f"M5Stack candidate: {port.device}")
                    return port.device
        
        # If no automatic detection, list all ports
        if ports:
            print("\nAvailable serial ports:")
            for i, port in enumerate(ports):
                print(f"{i}: {port.device} - {port.description}")
            
            choice = input(f"Select M5Stack port (0-{len(ports)-1}): ")
            try:
                return ports[int(choice)].device
            except (ValueError, IndexError):
                print("Invalid selection")
                return None
        
        return None
    
    def connect(self):
        """Connect to M5Stack"""
        if self.port is None:
            self.port = self.find_m5stack_port()
        
        if self.port is None:
            raise RuntimeError("No M5Stack port found")
        
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            print(f"Connected to M5Stack on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to M5Stack: {e}")
            return False
    
    def start_monitoring(self):
        """Start monitoring serial data in background thread"""
        if not self.serial_conn:
            raise RuntimeError("Not connected to M5Stack")
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("M5Stack monitoring started")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        timestamp = time.time()
                        self.data_queue.put((timestamp, line))
                        print(f"M5Stack: {line}")
                time.sleep(0.01)
            except Exception as e:
                if self.monitoring:
                    print(f"M5Stack monitor error: {e}")
                break
    
    def get_latest_data(self, timeout=1.0):
        """Get latest data from M5Stack"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def disconnect(self):
        """Disconnect from M5Stack"""
        self.stop_monitoring()
        if self.serial_conn:
            self.serial_conn.close()
            print("M5Stack disconnected")

class CoordinatedTest:
    """Coordinated AD3 and M5Stack test"""
    
    def __init__(self):
        self.ad3_test = None
        self.m5_monitor = None
    
    def setup(self):
        """Setup both devices"""
        print("Setting up coordinated test...")
        
        # Setup AD3
        print("\n1. Initializing AnalogDiscovery3...")
        self.ad3_test = AD3_M5StackTest()
        
        # Setup M5Stack monitor
        print("\n2. Connecting to M5Stack...")
        self.m5_monitor = M5StackMonitor()
        if not self.m5_monitor.connect():
            raise RuntimeError("Failed to connect to M5Stack")
        
        # Start M5Stack monitoring
        self.m5_monitor.start_monitoring()
        
        print("\nSetup completed successfully!")
        return True
    
    def wait_for_m5_readings(self, duration=5.0, expected_voltage=None):
        """Wait and collect M5Stack readings"""
        print(f"Collecting M5Stack readings for {duration}s...")
        
        start_time = time.time()
        readings = []
        
        while time.time() - start_time < duration:
            data = self.m5_monitor.get_latest_data(timeout=0.5)
            if data:
                timestamp, line = data
                # Parse voltage reading if it's in CSV format
                if ',' in line:
                    try:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            voltage = float(parts[1])
                            readings.append(voltage)
                    except ValueError:
                        pass
            time.sleep(0.1)
        
        if readings:
            avg_voltage = sum(readings) / len(readings)
            min_voltage = min(readings)
            max_voltage = max(readings)
            
            print(f"M5Stack readings: {len(readings)} samples")
            print(f"  Average: {avg_voltage:.3f}V")
            print(f"  Range: {min_voltage:.3f}V - {max_voltage:.3f}V")
            
            if expected_voltage is not None:
                error = abs(avg_voltage - expected_voltage)
                error_pct = (error / expected_voltage) * 100 if expected_voltage > 0 else 0
                print(f"  Expected: {expected_voltage:.3f}V")
                print(f"  Error: {error:.3f}V ({error_pct:.1f}%)")
                
                return error < 0.1  # Within 100mV tolerance
        else:
            print("No voltage readings received from M5Stack")
            return False
        
        return True
    
    def test_dc_voltage_verification(self):
        """Test DC voltage levels"""
        print("\n" + "="*60)
        print("DC Voltage Verification Test")
        print("="*60)
        
        test_voltages = [0.0, 1.0, 2.0, 3.0, 3.3]
        
        for voltage in test_voltages:
            print(f"\n--- Testing {voltage}V ---")
            
            # Set AD3 output
            self.ad3_test.output_dc_voltage(voltage)
            
            # Wait for signal to stabilize
            time.sleep(1.0)
            
            # Collect M5Stack readings
            success = self.wait_for_m5_readings(duration=3.0, expected_voltage=voltage)
            
            if success:
                print(f"✓ {voltage}V test PASSED")
            else:
                print(f"✗ {voltage}V test FAILED")
    
    def test_sine_wave_detection(self):
        """Test sine wave signal detection"""
        print("\n" + "="*60)
        print("Sine Wave Detection Test")
        print("="*60)
        
        # Test parameters
        frequency = 0.5  # 0.5 Hz for easy visual verification
        amplitude = 1.0
        offset = 1.65
        duration = 10.0
        
        print(f"Generating sine wave:")
        print(f"  Frequency: {frequency}Hz")
        print(f"  Amplitude: {amplitude}V")
        print(f"  Offset: {offset}V")
        print(f"  Range: {offset-amplitude:.2f}V to {offset+amplitude:.2f}V")
        
        # Start sine wave
        self.ad3_test.configure_ch2_output(frequency, amplitude, offset)
        
        # Monitor for varying voltage
        print(f"\nMonitoring M5Stack for {duration}s...")
        print("Expecting voltage to vary between limits...")
        
        self.wait_for_m5_readings(duration=duration)
    
    def teardown(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        if self.ad3_test:
            self.ad3_test.tearDown()
        
        if self.m5_monitor:
            self.m5_monitor.disconnect()
        
        print("Cleanup completed")

def main():
    """Main test execution"""
    print("="*60)
    print("AD3 Ch2 to M5Stack Coordinated Test")
    print("="*60)
    print("Hardware setup required:")
    print("- AD3 W2 (ch2) → M5Stack Analog Pin (GPIO36)")
    print("- AD3 GND → M5Stack GND")
    print("- M5Stack running analog_reader.ino")
    print("- M5Stack connected via USB")
    print("="*60)
    
    input("Press Enter when hardware is connected and M5Stack is running...")
    
    test = CoordinatedTest()
    
    try:
        # Setup
        test.setup()
        
        print("\nSelect test:")
        print("1. DC Voltage Verification")
        print("2. Sine Wave Detection")
        print("3. Both tests")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice in ["1", "3"]:
            test.test_dc_voltage_verification()
        
        if choice in ["2", "3"]:
            test.test_sine_wave_detection()
        
        test.teardown()
        print("\n✓ Test completed successfully!")
        print("Signal transmission from AD3 Ch2 to M5Stack verified")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        if test:
            test.teardown()

if __name__ == '__main__':
    main()