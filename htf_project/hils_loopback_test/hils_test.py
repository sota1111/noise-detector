#!/usr/bin/env python3
"""
HILS (Hardware-in-the-Loop Simulation) AnalogDiscovery3 Loopback Test
Tests signal generation on ch1 and measurement on ai[0] for multiple frequencies
"""

import time
import numpy as np
from analog_discovery_plug import AnalogDiscoveryPlug

def main():
    """Run HILS loopback test"""
    print("="*60)
    print("HILS AnalogDiscovery3 Loopback Test")
    print("="*60)
    
    try:
        # Initialize device
        print("Initializing AnalogDiscovery3...")
        device = AnalogDiscoveryPlug()
        
        # Test single frequency
        print("\nTesting loopback at 1000Hz...")
        result = device.loopback_test(1000.0, 1.0, 1.0)
        
        print(f"\nResults:")
        print(f"  Input: {result['input_frequency']}Hz, {result['input_amplitude']}V")
        print(f"  Measured: {result['measured_frequency']:.1f}Hz, {result['measured_amplitude']:.3f}V")
        print(f"  Frequency Error: {result['frequency_error']:.2f}%")
        print(f"  Amplitude Error: {result['amplitude_error']:.2f}%")
        
        # Test multiple frequencies
        print("\nTesting multiple frequencies...")
        results = device.multi_frequency_test()
        
        print(f"\nSummary:")
        print(f"{'Frequency':<10} {'Freq Error':<12} {'Amp Error':<12} {'Status':<8}")
        print("-" * 50)
        
        for freq, result in results.items():
            freq_err = result['frequency_error']
            amp_err = result['amplitude_error']
            status = "PASS" if freq_err < 5.0 and amp_err < 10.0 else "FAIL"
            print(f"{freq:<10} {freq_err:<12.2f} {amp_err:<12.2f} {status:<8}")
        
        device.tearDown()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()