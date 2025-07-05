# Thermal Violations Test Network

## Overview
This network is designed to test the LLM agent's ability to resolve **thermal violations** (line overloading scenarios).

## Network Characteristics
- **Base Network**: IEEE 14-bus system
- **Total Buses**: 14
- **Primary Violation Type**: Thermal violations (line overloading)
- **Violations**: 3 thermal violations, 4 voltage violations (total: 7)

## Violation Configuration
### Thermal Violations
- Line capacity limits reduced to create bottlenecks:
  - Lines from bus 1: `max_i_ka = 0.15` (very low limit)
  - Lines from bus 4: `max_i_ka = 0.20` (bottleneck)
  - Lines to bus 5: `max_i_ka = 0.18` (constraint)

### Load Configuration
- High loads to stress the system:
  - Bus 4: 0.8 MW
  - Bus 5: 0.9 MW  
  - Bus 9: 0.6 MW

## Available Resources
- **Curtailable Loads**: 3 (buses 4, 5, 9)
- **Switches**: 3 for reconfiguration
- **Max Batteries**: 3 (agent limit)
- **Resource Ratio**: 1.3 resources per violation

## Solution Strategy
The agent should prioritize:
1. **Switch reconfigurations** to reroute power and reduce line loading
2. **Load curtailment** on heavily loaded buses (4, 5, 9)
3. **Strategic battery placement** for voltage support

## Configurable Parameters
To modify violation severity, adjust:
- `net.line.max_i_ka` values (lower = more violations)
- Load `p_mw` values (higher = more stress)
- Switch positions for different network topology

## Usage
```bash
cd data/networks/test_thermal_violations
python generate.py  # Regenerate network
```