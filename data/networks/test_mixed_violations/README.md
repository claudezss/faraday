# Mixed Violations Test Network

## Overview
This network is designed to test the LLM agent's ability to resolve **both voltage and thermal violations** simultaneously - a complex optimization scenario.

## Network Characteristics
- **Base Network**: CIGRE MV network with extensions
- **Total Buses**: 15
- **Violation Types**: Mixed (both voltage and thermal)
- **Violations**: 10 voltage violations, 4 thermal violations (total: 14)

## Violation Configuration
### Thermal Violations
- Reduced line capacities create bottlenecks:
  - Lines from bus 2: `max_i_ka = 0.15`
  - Lines from bus 6: `max_i_ka = 0.18`
  - Lines to bus 9: `max_i_ka = 0.12`

### Voltage Violations
- High loads cause voltage drops:
  - Bus 7: 2.5 MW (high load)
  - Bus 9: 2.2 MW (high load)
  - Bus 11: 2.0 MW (high load)
  - Bus 12: 1.8 MW (moderate load)

### Distributed Generation
- End-bus generation adds complexity:
  - Bus 13: 1.2 MW PV (may cause overvoltage)
  - Bus 14: 0.8 MW Wind (voltage support)

## Available Resources
- **Curtailable Loads**: 3 (buses 7, 9, 11)
- **Switches**: 12 for complex reconfiguration
- **DG Units**: 11 (significant flexibility)
- **Max Batteries**: 3 (agent limit)
- **Resource Ratio**: 1.3 resources per violation

## Solution Strategy
The agent should consider:
1. **Coordinated approach** addressing both violation types
2. **Switch optimization** for thermal relief and voltage improvement
3. **Strategic battery placement** for voltage support
4. **Load curtailment** on key buses affecting multiple violations
5. **DG coordination** for voltage management

## Optimization Opportunities
This network is ideal for testing:
- Multi-objective optimization algorithms
- Coordinated action planning
- Resource allocation strategies
- Violation clustering techniques

## Configurable Parameters
To modify violation patterns:
- **Thermal**: Adjust `max_i_ka` values (lower = more thermal violations)
- **Voltage**: Modify load `p_mw` values (higher = more voltage violations)
- **Complexity**: Change DG output levels and switch positions
- **Topology**: Add/remove connections to test different scenarios

## Usage
```bash
cd data/networks/test_mixed_violations
python generate.py  # Regenerate network
```

## Advanced Testing
This network can be used to validate:
- Action optimization algorithms
- Violation clustering effectiveness
- Multi-violation coordination strategies
- Resource allocation efficiency