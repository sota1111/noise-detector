#!/usr/bin/env python3
"""
AnalogDiscovery3 Ch2 to M5Stack Analog Input Test
Tests signal transmission from AD3 ch2 (max 3.3V) to M5Stack analog port
"""

import time
import numpy as np
from pydwf import DwfLibrary, DwfDevice, DeviceEnumeration, DwfState, DwfAnalogOutFunction, DwfAnalogOutNode

class AD3_M5StackTest:
    """Test AD3 ch2 output to M5Stack analog input"""
    
    def __init__(self):
        """Initialize AnalogDiscovery3"""
        self.dwf_lib = DwfLibrary()
        self.device = None
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize AD3 device"""
        enumeration = DeviceEnumeration(self.dwf_lib)
        device_count = enumeration.enumerateDevices()
        print(f"Found {device_count} device(s)")
        
        if device_count == 0:
            raise RuntimeError("No AnalogDiscovery device found")
        
        # Open first device
        self.device = self.dwf_lib.deviceControl.open(0)
        print("AnalogDiscovery3 opened successfully")
    
    def configure_ch2_output(self, frequency=1000.0, amplitude=1.5, offset=1.65):
        """Configure ch2 output with 3.3V limit
        
        Args:
            frequency: Output frequency in Hz
            amplitude: Peak amplitude in V (max 1.65V for 3.3V total)
            offset: DC offset in V (1.65V centers signal at 1.65V)
        """
        max_voltage = amplitude + offset
        if max_voltage > 3.3:
            raise ValueError(f"Max voltage {max_voltage}V exceeds 3.3V limit")
        
        print(f"Configuring ch2: freq={frequency}Hz, amp={amplitude}V, offset={offset}V")
        print(f"Voltage range: {offset-amplitude:.2f}V to {offset+amplitude:.2f}V")
        
        # Configure analog output channel 1 (ch2)
        self.device.analogOut.nodeEnableSet(1, DwfAnalogOutNode.Carrier, True)
        self.device.analogOut.nodeFunctionSet(1, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.Sine)
        self.device.analogOut.nodeFrequencySet(1, DwfAnalogOutNode.Carrier, frequency)
        self.device.analogOut.nodeAmplitudeSet(1, DwfAnalogOutNode.Carrier, amplitude)
        self.device.analogOut.nodeOffsetSet(1, DwfAnalogOutNode.Carrier, offset)
        
        # Start output
        self.device.analogOut.configure(1, True)
        print("Ch2 output started")
    
    def output_dc_voltage(self, voltage):
        """Output constant DC voltage on ch2
        
        Args:
            voltage: DC voltage (0-3.3V)
        """
        if voltage > 3.3 or voltage < 0:
            raise ValueError(f"Voltage {voltage}V must be between 0-3.3V")
        
        print(f"Setting ch2 DC output to {voltage}V")
        
        # Configure as DC
        self.device.analogOut.nodeEnableSet(1, DwfAnalogOutNode.Carrier, True)
        self.device.analogOut.nodeFunctionSet(1, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.DC)
        self.device.analogOut.nodeOffsetSet(1, DwfAnalogOutNode.Carrier, voltage)
        
        # Start output
        self.device.analogOut.configure(1, True)
        print(f"Ch2 DC output at {voltage}V started")
    
    def voltage_sweep_test(self, start_v=0.0, end_v=3.3, steps=10, step_duration=2.0):
        """Perform voltage sweep test for M5Stack verification
        
        Args:
            start_v: Starting voltage
            end_v: Ending voltage  
            steps: Number of steps
            step_duration: Duration of each step in seconds
        """
        print(f"\nStarting voltage sweep: {start_v}V to {end_v}V in {steps} steps")
        print("Monitor M5Stack serial output for voltage readings")
        print("=" * 60)
        
        voltages = np.linspace(start_v, end_v, steps)
        
        for i, voltage in enumerate(voltages):
            print(f"Step {i+1}/{steps}: Setting {voltage:.2f}V")
            self.output_dc_voltage(voltage)
            
            print(f"Waiting {step_duration}s for M5Stack to read...")
            time.sleep(step_duration)
        
        print("\nVoltage sweep completed")
    
    def sine_wave_test(self, frequency=1.0, amplitude=1.0, offset=1.65, duration=10.0):
        """Generate sine wave for M5Stack monitoring
        
        Args:
            frequency: Sine wave frequency in Hz
            amplitude: Peak amplitude in V
            offset: DC offset in V
            duration: Test duration in seconds
        """
        print(f"\nStarting sine wave test:")
        print(f"Frequency: {frequency}Hz, Amplitude: {amplitude}V, Offset: {offset}V")
        print(f"Duration: {duration}s")
        print("Monitor M5Stack serial for changing voltage readings")
        print("=" * 60)
        
        self.configure_ch2_output(frequency, amplitude, offset)
        
        print("Sine wave output active...")
        time.sleep(duration)
        
        print("Sine wave test completed")
    
    def stop_output(self):
        """Stop ch2 output"""
        if self.device:
            self.device.analogOut.configure(1, False)
            print("Ch2 output stopped")
    
    def tearDown(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.stop_output()
        if self.device:
            try:
                self.device.close()
                print("Device closed successfully")
            except Exception as e:
                print(f"Error during cleanup: {e}")

def main():
    """Main test execution"""
    print("=" * 60)
    print("AnalogDiscovery3 Ch2 to M5Stack Test")
    print("=" * 60)
    print("Hardware setup:")
    print("- AD3 W2 (ch2) → M5Stack Analog Pin")
    print("- AD3 GND → M5Stack GND")
    print("- M5Stack connected via USB for serial monitoring")
    print("=" * 60)
    
    try:
        test = AD3_M5StackTest()
        
        print("\nRunning automatic DC voltage sweep test...")
        print("Monitor M5Stack LCD and serial output for voltage readings")
        
        # DC voltage sweep test
        test.voltage_sweep_test(start_v=0.0, end_v=3.3, steps=11, step_duration=3.0)
        
        test.tearDown()
        print("\nTest completed successfully!")
        print("Check M5Stack serial output to verify signal reception")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()