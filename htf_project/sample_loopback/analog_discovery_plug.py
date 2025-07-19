# analog_discovery_plug.py
"""Analog Discovery plug for OpenHTF testing framework"""

import json
import os
import dwfpy as dwf
from openhtf import plugs
import numpy as np
from .analog_output import AnalogOutputFactory


class AnalogDiscoveryPlug(plugs.BasePlug):
    """OpenHTF plug for Analog Discovery device operations"""
    
    def __init__(self, config_path=None):
        """Initialize Analog Discovery device
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 
                                     '../tests/analog/test_config.json')
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize device
        devices = list(dwf.Device.enumerate())
        print(f"DEBUG: Found {len(devices)} Analog Discovery devices")
        if not devices:
            raise RuntimeError("No Analog Discovery device found.")
        
        self.serial = devices[0].serial_number
        print(f"Using device with serial: {self.serial}")

    def measure_sine_waveform(self):
        """Measure sine wave loopback with configured parameters
        
        Returns:
            numpy.ndarray: Measured waveform samples
        """
        config = self.config['tests']['sine_wave']
        device_config = self.config['device']
        
        print("Starting sine wave measurement...")
        
        with dwf.Device(serial_number=self.serial) as device:
            # Generate sine wave output
            generator = AnalogOutputFactory.create_generator('sine', device)
            generator.generate_waveform(
                frequency=config['output_frequency'],
                amplitude=config['output_amplitude'],
                offset=0.0
            )

            # Configure analog input and acquire data
            ai = device.analog_input
            ai[0].setup(range=device_config['input_range'])
            ai.single(
                sample_rate=config['sample_rate'], 
                buffer_size=config['buffer_size'], 
                configure=True, 
                start=True
            )
            
            samples = ai[0].get_data()
            print(f"Acquired {len(samples)} samples")
            return samples

    def measure_step_waveform(self):
        """Measure step wave loopback with configured parameters
        
        Returns:
            tuple: (all_samples, voltage_at_1s, voltage_at_3s)
        """
        config = self.config['tests']['step_wave']
        device_config = self.config['device']
        
        print("Starting step wave measurement...")
        
        with dwf.Device(serial_number=self.serial) as device:
            # Configure analog input
            ai = device.analog_input
            ai[0].setup(range=device_config['input_range'])

            all_samples = []
            
            # Generate step wave and collect samples
            segments = config['segments']
            for seg in range(segments):
                voltage = (config['low_voltage'] if seg < segments // 2 
                          else config['high_voltage'])
                
                print(f"Segment {seg + 1}/{segments}: outputting {voltage}V")
                
                # Set DC output
                device.analog_output['ch1'].setup(
                    function='dc', offset=voltage, start=True)
                
                # Sample for specified duration
                ai.single(
                    sample_rate=config['sample_rate'],
                    buffer_size=config['buffer_size'],
                    configure=True,
                    start=True
                )
                all_samples.extend(ai[0].get_data())

            # Extract specific time points (1s and 3s)
            sample_rate = config['sample_rate']
            v1 = all_samples[int(1.0 * sample_rate)]  # 1 second
            v3 = all_samples[int(3.0 * sample_rate)]  # 3 seconds
            
            print(f"Voltage at 1s: {v1:.3f}V, at 3s: {v3:.3f}V")
            print(f"Total samples collected: {len(all_samples)}")
            
            return all_samples, v1, v3

    def tearDown(self):
        """Clean up resources"""
        print("Analog Discovery plug teardown completed")
        pass
