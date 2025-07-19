# analog_output.py
"""Analog output waveform generation classes"""

import time
import numpy as np

class AnalogOutputGenerator:
    """Base class for analog output waveform generation"""
    
    def __init__(self, device):
        """Initialize with dwf device instance"""
        self.device = device
    
    def generate_waveform(self, **kwargs):
        """Generate waveform - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_waveform")


class SineWaveGenerator(AnalogOutputGenerator):
    """Sine wave generator"""
    
    def generate_waveform(self, frequency=1e3, amplitude=1.0, offset=0.0):
        """Generate sine wave output
        
        Args:
            frequency: Frequency in Hz
            amplitude: Amplitude in V
            offset: DC offset in V
        """
        print(f"Generating sine wave: freq={frequency}Hz, amp={amplitude}V, offset={offset}V")
        self.device.analog_output['ch1'].setup(
            function='sine',
            frequency=frequency,
            amplitude=amplitude,
            offset=offset,
            start=True
        )


class StepWaveGenerator(AnalogOutputGenerator):
    """Step wave generator"""
    
    def generate_waveform(self, segments=4, segment_duration=1.0, 
                         low_voltage=0.0, high_voltage=1.0):
        """Generate step wave output with multiple segments
        
        Args:
            segments: Number of segments
            segment_duration: Duration of each segment in seconds
            low_voltage: Low voltage level
            high_voltage: High voltage level
        """
        print(f"Generating step wave: {segments} segments, {segment_duration}s each")
        print(f"Voltage levels: low={low_voltage}V, high={high_voltage}V")
        
        for seg in range(segments):
            voltage = low_voltage if seg < segments // 2 else high_voltage
            print(f"Segment {seg + 1}/{segments}: {voltage}V")
            self.device.analog_output['ch1'].setup(
                function='dc', offset=voltage, start=True)
            time.sleep(segment_duration)


class AnalogOutputFactory:
    """Factory class for creating analog output generators"""
    
    @staticmethod
    def create_generator(generator_type, device):
        """Create appropriate generator based on type
        
        Args:
            generator_type: 'sine' or 'step'
            device: dwf device instance
            
        Returns:
            AnalogOutputGenerator instance
        """
        if generator_type == 'sine':
            return SineWaveGenerator(device)
        elif generator_type == 'step':
            return StepWaveGenerator(device)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")