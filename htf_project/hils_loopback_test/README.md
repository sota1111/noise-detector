# HILS AnalogDiscovery3 Loopback Test

Hardware-in-the-Loop Simulation (HILS) test using AnalogDiscovery3 for signal validation.

## Overview

Tests signal generation on ch1 and measurement on ai[0] to validate analog signal path accuracy across multiple frequencies.

## Features

- **Multi-frequency testing**: 100Hz, 500Hz, 1kHz, 2kHz
- **Loopback validation**: Verifies output signal matches input measurement  
- **Error analysis**: Calculates frequency and amplitude errors with PASS/FAIL criteria
- **Hardware validation**: Real AnalogDiscovery3 device testing

## Quick Start

1. **Hardware Setup**:
   ```
   AnalogDiscovery3 connections:
   - W1 (ch1 output) → 1+ (ai[0] input)
   - W1 GND → 1- (ai[0] ground)  
   - USB to computer
   ```

2. **Run Test**:
   ```bash
   cd /home/sota/dev/2025/hils/noise-detector
   source htf_env/bin/activate
   cd htf_project/hils_loopback_test
   python hils_test.py
   ```

## Verification

Check setup before testing:
```bash
python setup_test.py
```

## Configuration

Edit `test_config.json` for:
- Test frequencies: `[100, 500, 1000, 2000]` Hz
- Signal amplitude: `1.0` V
- Sample rates and tolerances

## Expected Results

```
Frequency  Freq Error   Amp Error    Status  
--------------------------------------------------
100Hz      0.00%        0.15%        PASS    
500Hz      0.00%        0.07%        PASS    
1000Hz     0.00%        0.24%        PASS    
2000Hz     0.00%        1.46%        PASS    
```

## Requirements

- AnalogDiscovery3 device
- pydwf library
- numpy