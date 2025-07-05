# Large-Scale Test Network (118+ nodes)

## Overview
This network is designed to test **large-scale optimization** and **advanced token reduction** algorithms using the IEEE 118-bus system.

## Network Characteristics
- **Base Network**: IEEE 118-bus transmission system
- **Total Buses**: 118 (large-scale test case)
- **Primary Purpose**: Token explosion scenarios and Phase 3 algorithm validation
- **Violations**: 8 voltage violations (solvable with conservative violations)

## Network Architecture
### IEEE 118 System
- Large transmission network with complex topology
- 173 transmission lines
- 99 load buses
- Realistic large-scale power system structure

### Extensions Added
- **Distributed Generation**: 11 units (1-5 MW each)
- **Load Curtailment**: 19 curtailable loads (~20% of total)
- **Network Switches**: 7 reconfiguration switches

## Violation Configuration
### Conservative Approach
- **Thermal violations**: Very light capacity reductions (15% reduction)
- **Voltage violations**: Minimal load increases (5-10%)
- **Solvable design**: High resource-to-violation ratio (3.6:1)

## Phase 3 Testing Capabilities
### Token Reduction Validation
- **Standard representation**: ~30,888 characters
- **Violations-only**: 842 characters (**97.3% reduction**)
- **Hierarchical**: 3,041 characters (**90.2% reduction**)
- **Advanced compact**: 3,171 characters (**89.7% reduction**)

### Hierarchical Features
- **4 electrical zones** for partitioning
- **Zone-based violation analysis**
- **Multi-level representation switching**

## Advanced Testing Scenarios
This network is ideal for:
1. **Token explosion testing** - demonstrates why Phase 3 is essential
2. **Large-scale action optimization** - coordination across many buses
3. **Hierarchical zone partitioning** - realistic electrical distance calculations
4. **Performance scalability** - real-world complexity without overwhelming violations

## Solution Strategy
### Resource Management
- 19 curtailable loads distributed across network
- 7 switches for strategic reconfiguration
- 11 DG units for voltage support
- 3 strategic battery placements

### Optimization Challenges
- **Scale**: 118 buses approach LLM token limits
- **Complexity**: Realistic transmission system topology
- **Coordination**: Multiple resource types requiring strategic planning

## Performance Characteristics
### Token Reduction Critical
- Standard representation exceeds practical LLM limits
- Phase 3 algorithms **essential** for effective planning
- Demonstrates real-world scalability challenges

### Solvability
- Conservative violations ensure solvability
- Focus on algorithm testing rather than problem difficulty
- High resource availability for solution flexibility

## Usage
```bash
cd data/networks/test_massive_2000
python generate.py  # Regenerate network
```

## Key Benefits
✅ **Real-world scale** - Approaches practical LLM limits  
✅ **Token reduction validation** - 90%+ compression demonstrated  
✅ **Phase 3 algorithm testing** - Essential for this scale  
✅ **Hierarchical partitioning** - Multi-zone analysis  
✅ **Performance benchmarking** - Scalability validation

## Notes
- Network named "massive_2000" but uses IEEE 118 for stability
- Focus on token reduction rather than raw node count
- Demonstrates why advanced algorithms become essential at scale
- Excellent test case for Phase 3 feature validation