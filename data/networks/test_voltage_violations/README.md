# Voltage Violations Test Network

## Overview
This network is designed to test the LLM agent's ability to resolve **voltage violations** (under/over voltage scenarios).

## Network Characteristics
- **Base Network**: CIGRE MV network
- **Total Buses**: 15
- **Primary Violation Type**: Voltage violations (undervoltage conditions)
- **Violations**: 11 voltage violations, 6 thermal violations (total: 17)

## Violation Configuration
### Voltage Violations
- High loads at end of feeders cause voltage drops:
  - Bus 7: 4.0 MW (very high load)
  - Bus 9: 3.5 MW (high load)
  - Bus 11: 3.0 MW (high load)
  - Bus 12: 2.5 MW (moderate load)

### Line Impedance
- Increased resistance to create voltage bottlenecks:
  - Lines to bus 7: `r_ohm_per_km × 2.0`
  - Lines to bus 9: `r_ohm_per_km × 1.8`
  - Lines to bus 11: `r_ohm_per_km × 1.5`

### Distributed Generation
- PV units that may cause voltage complexity:
  - Bus 13: 0.8 MW PV
  - Bus 14: 0.6 MW PV

## Available Resources
- **Curtailable Loads**: 3 (buses 7, 9, 11)
- **Switches**: 11 for reconfiguration
- **DG Units**: 2 (potential voltage support)
- **Max Batteries**: 3 (agent limit)
- **Resource Ratio**: 1.0 resources per violation

## Solution Strategy
The agent should prioritize:
1. **Load curtailment** at high-load buses (7, 9, 11)
2. **Battery placement** for voltage support near violations
3. **Switch reconfiguration** to improve voltage profiles

## Configurable Parameters
To modify violation severity, adjust:
- Load `p_mw` values (higher = more undervoltage)
- Line resistance multipliers (higher = more voltage drop)
- DG output levels (higher = potential overvoltage)

## Usage
```bash
cd data/networks/test_voltage_violations
python generate.py  # Regenerate network
```