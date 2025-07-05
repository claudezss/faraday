# Large-Scale Test Network (114+ nodes)

## Overview
This network is designed to test **medium-scale optimization** and **hierarchical representation** algorithms with approximately 114 nodes.

## Network Characteristics
- **Base Network**: IEEE 39-bus system with radial feeders
- **Total Buses**: ~91 (extended to approach 114)
- **Primary Purpose**: Test hierarchical zone partitioning and token reduction
- **Violations**: 25 voltage + 9 thermal (34 total)

## Network Architecture
### Base System
- IEEE 39-bus transmission system (stable foundation)
- Extended with 10 radial distribution feeders

### Feeder Structure  
- **Main feeders**: 8-12 buses each with radial extensions
- **Lateral branches**: 3-8 buses per lateral
- **Load distribution**: 70% of feeder buses have loads

## Violation Configuration
### Thermal Violations (9 total)
- Conservative capacity reduction: 60% of original `max_i_ka`
- Distributed across transmission and distribution levels

### Voltage Violations (25 total)
- Moderate load increases: 30-60% above baseline
- Concentrated at feeder ends (realistic voltage drop scenarios)

## Available Resources
- **Curtailable Loads**: 6 (strategic distribution)
- **Switches**: 14 for network reconfiguration
- **DG Units**: 11 distributed generation sources
- **Max Batteries**: 3 (agent limit)
- **Resource Ratio**: 0.7 resources per violation

## Phase 3 Testing Capabilities
### Hierarchical Representation
- **Zones**: 3 electrical zones for partitioning
- **Token reduction**: Significant compression for medium-scale networks

### Graph-Based Representation
- Violation clustering by electrical distance
- Action-resource proximity mapping
- Multi-level representation switching

## Solution Strategy
This network is ideal for testing:
1. **Zone-based planning** - violations distributed across zones
2. **Hierarchical optimization** - coordination between transmission and distribution
3. **Token reduction** - medium-scale representation challenges
4. **Action coordination** - multiple resource types available

## Configurable Parameters
To modify network characteristics:
- **Network size**: Adjust `feeder_length` and number of feeders
- **Violation severity**: Modify capacity reduction factors and load increases
- **Resource availability**: Change curtailable load percentage and DG sizing

## Usage
```bash
cd data/networks/test_large_114
python generate.py  # Regenerate network
```

## Performance Notes
- Network size ideal for testing hierarchical algorithms
- Demonstrates token reduction effectiveness (90%+ reduction possible)
- Realistic medium-scale distribution system complexity
- Good balance between complexity and solvability