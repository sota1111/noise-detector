#!/usr/bin/env python3
"""
Setup script to verify AnalogDiscovery3 connectivity and pydwf installation
"""

import sys
import subprocess

def check_pydwf():
    """Check if pydwf is installed and working"""
    try:
        from pydwf import DwfLibrary, DeviceEnumeration
        print("✓ pydwf library imported successfully")
        
        # Initialize library
        dwf_lib = DwfLibrary()
        print("✓ DWF library initialized")
        
        # Check for devices
        enumeration = DeviceEnumeration(dwf_lib)
        device_count = enumeration.enumerateDevices()
        print(f"✓ Found {device_count} AnalogDiscovery device(s)")
        
        if device_count == 0:
            print("⚠ Warning: No AnalogDiscovery devices found")
            print("  Make sure device is connected and drivers are installed")
            print("  (Software API is working correctly - this is OK for development)")
        
        # Get device info
        for i in range(device_count):
            device_name = enumeration.deviceName(i)
            serial_number = enumeration.serialNumber(i)
            print(f"  Device {i}: {device_name} (Serial: {serial_number})")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import pydwf: {e}")
        print("  Install with: pip install pydwf")
        return False
    except Exception as e:
        print(f"✗ Error with pydwf: {e}")
        return False

def check_numpy():
    """Check if numpy is installed"""
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        print("  Install with: pip install numpy")
        return False

def install_requirements():
    """Install requirements if missing"""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def main():
    """Main setup verification"""
    print("AnalogDiscovery3 HILS Test Setup Verification")
    print("=" * 50)
    
    all_good = True
    
    # Check numpy
    if not check_numpy():
        all_good = False
    
    # Check pydwf
    if not check_pydwf():
        all_good = False
    
    if not all_good:
        print("\n" + "!" * 50)
        print("Some dependencies are missing. Installing requirements...")
        if install_requirements():
            print("\nRe-checking after installation...")
            all_good = check_numpy() and check_pydwf()
    
    print("\n" + "=" * 50)
    if all_good:
        print("✓ Setup verification PASSED")
        print("  You can now run the HILS test: python hils_test.py")
    else:
        print("✗ Setup verification FAILED")
        print("  Please resolve the issues above before running tests")
        sys.exit(1)

if __name__ == '__main__':
    main()