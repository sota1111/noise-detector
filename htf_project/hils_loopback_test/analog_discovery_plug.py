# analog_discovery_plug.py
"""Analog Discovery device control using pydwf"""

import json
import os
import time
import numpy as np
from pydwf import DwfLibrary, DwfDevice, DeviceEnumeration, DwfState, DwfAnalogOutFunction, DwfAcquisitionMode, DwfAnalogOutNode


class AnalogDiscoveryPlug:
    """Analog Discovery device control using pydwf"""
    
    def __init__(self, config_path=None):
        """Initialize Analog Discovery device
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize DWF library
        self.dwf_lib = DwfLibrary()
        print("DWF library initialized successfully")
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'test_config.json')
        
        self.config = self._load_or_create_config(config_path)
        
        # Initialize device
        self.device = None
        self._initialize_device()
        
    def _load_or_create_config(self, config_path):
        """Load configuration or create default if not exists"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            default_config = {
                "device": {
                    "input_range": 5.0,
                    "output_range": 5.0
                },
                "tests": {
                    "sine_wave": {
                        "output_frequency": 1000.0,
                        "output_amplitude": 1.0,
                        "sample_rate": 10000.0,
                        "buffer_size": 1000,
                        "duration": 1.0
                    },
                    "loopback": {
                        "frequencies": [100, 500, 1000, 2000],
                        "amplitude": 1.0,
                        "sample_rate": 20000.0,
                        "buffer_size": 2000,
                        "settling_time": 0.1
                    }
                }
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created default config at {config_path}")
            return default_config

    def _initialize_device(self):
        """Initialize Analog Discovery device"""
        # Enumerate devices
        enumeration = DeviceEnumeration(self.dwf_lib)
        device_count = enumeration.enumerateDevices()
        print(f"Found {device_count} device(s)")
        
        if device_count == 0:
            raise RuntimeError("No Analog Discovery device found")
        
        # Get device info
        for i in range(device_count):
            device_name = enumeration.deviceName(i)
            serial_number = enumeration.serialNumber(i)
            print(f"Device {i}: {device_name} (Serial: {serial_number})")
        
        # Open first device
        self.device = self.dwf_lib.deviceControl.open(0)  # Open device 0
        print("Device opened successfully")

    def loopback_test(self, frequency=1000.0, amplitude=1.0, duration=1.0):
        """Perform loopback test with ch1 output to ai[0] input
        
        Args:
            frequency: Output frequency in Hz
            amplitude: Output amplitude in V
            duration: Test duration in seconds
            
        Returns:
            dict: Test results including input/output data
        """
        if not self.device:
            raise RuntimeError("Device not initialized")
            
        config = self.config['tests']['loopback']
        sample_rate = config['sample_rate']
        buffer_size = int(sample_rate * duration)
        
        print(f"Starting loopback test: freq={frequency}Hz, amp={amplitude}V, duration={duration}s")
        
        try:
            # Configure analog output (ch1)
            print("Configuring analog output...")
            self.device.analogOut.nodeEnableSet(0, DwfAnalogOutNode.Carrier, True)  # Channel 0, carrier node
            self.device.analogOut.nodeFunctionSet(0, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.Sine)
            self.device.analogOut.nodeFrequencySet(0, DwfAnalogOutNode.Carrier, frequency)
            self.device.analogOut.nodeAmplitudeSet(0, DwfAnalogOutNode.Carrier, amplitude)
            self.device.analogOut.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, 0.0)
            
            # Configure analog input (ai[0])
            print("Configuring analog input...")
            input_range = self.config['device']['input_range']
            self.device.analogIn.channelEnableSet(0, True)
            self.device.analogIn.channelRangeSet(0, input_range)
            self.device.analogIn.frequencySet(sample_rate)
            self.device.analogIn.bufferSizeSet(buffer_size)
            
            # Configure acquisition mode
            self.device.analogIn.acquisitionModeSet(DwfAcquisitionMode.Single)
            
            # Start output
            print("Starting analog output...")
            self.device.analogOut.configure(0, True)
            
            # Wait for settling
            time.sleep(config['settling_time'])
            
            # Start acquisition
            print("Starting data acquisition...")
            self.device.analogIn.configure(False, True)
            
            # Wait for acquisition to complete
            status = DwfState.Done
            while True:
                status = self.device.analogIn.status(True)
                if status == DwfState.Done:
                    break
                elif status == DwfState.Config or status == DwfState.Prefill or status == DwfState.Armed:
                    continue
                else:
                    time.sleep(0.001)
            
            # Read data
            print("Reading acquired data...")
            samples = self.device.analogIn.statusData(0, buffer_size)
            
            # Stop output
            self.device.analogOut.configure(0, False)
            
            # Analyze results
            time_array = np.linspace(0, duration, len(samples))
            
            # Calculate frequency from FFT
            fft = np.fft.fft(samples)
            freqs = np.fft.fftfreq(len(samples), 1/sample_rate)
            peak_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            measured_frequency = abs(freqs[peak_freq_idx])
            
            # Calculate amplitude (RMS to peak)
            measured_amplitude = np.sqrt(2) * np.std(samples)
            
            results = {
                'input_frequency': frequency,
                'input_amplitude': amplitude,
                'measured_frequency': measured_frequency,
                'measured_amplitude': measured_amplitude,
                'frequency_error': abs(measured_frequency - frequency) / frequency * 100,
                'amplitude_error': abs(measured_amplitude - amplitude) / amplitude * 100,
                'samples': samples.tolist(),
                'time_array': time_array.tolist(),
                'sample_rate': sample_rate,
                'buffer_size': buffer_size
            }
            
            print(f"Input: {frequency}Hz, {amplitude}V")
            print(f"Measured: {measured_frequency:.1f}Hz, {measured_amplitude:.3f}V")
            print(f"Errors: freq={results['frequency_error']:.2f}%, amp={results['amplitude_error']:.2f}%")
            
            return results
            
        except Exception as e:
            # Ensure output is stopped in case of error
            if self.device:
                self.device.analogOut.configure(0, False)
            raise e

    def multi_frequency_test(self):
        """Test multiple frequencies for HILS evaluation
        
        Returns:
            dict: Results for all test frequencies
        """
        config = self.config['tests']['loopback']
        frequencies = config['frequencies']
        amplitude = config['amplitude']
        
        print(f"Running multi-frequency test: {frequencies}")
        
        all_results = {}
        for freq in frequencies:
            print(f"\n--- Testing frequency: {freq}Hz ---")
            result = self.loopback_test(freq, amplitude, 1.0)
            all_results[freq] = result
            
            # Brief pause between tests
            time.sleep(0.5)
        
        return all_results

    def tearDown(self):
        """Clean up resources"""
        print("Cleaning up Analog Discovery resources...")
        if self.device:
            try:
                # Stop all outputs
                self.device.analogOut.configure(0, False)
                self.device.analogOut.configure(1, False)
                # Close device
                self.device.close()
                print("Device closed successfully")
            except Exception as e:
                print(f"Error during cleanup: {e}")
        print("Analog Discovery plug teardown completed")