# Faraday

AI-powered power grid optimization system that automatically detects and resolves electrical network violations.

## Overview

Faraday is an intelligent power grid optimization system that combines advanced AI agents with power system analysis to automatically identify and resolve electrical network violations. The system uses LangGraph-based workflows and pandapower simulation to provide both automated and interactive solutions for power grid management.

### Key Components

- **AI Agent Workflow**: LangGraph-based multi-agent system with planner, executor, and analysis agents
- **Power System Analysis**: Built on pandapower for accurate electrical network simulation
- **Violation Detection**: Automated identification of voltage and thermal violations with configurable thresholds
- **Smart Optimization**: AI-generated action plans that minimize interventions through coordinated strategies
- **Dual Interfaces**: Command-line tool for automation and Streamlit dashboard for interactive analysis

### Available Actions

- **Topology Control**: Switch operations to reconfigure network connectivity
- **Demand Management**: Load curtailment to reduce system stress
- **Storage Integration**: Battery addition for grid flexibility and support

### Architecture

The system follows a modular architecture with:
- **Agents** (`faraday/agents/`): AI workflow orchestration and planning
- **Tools** (`faraday/tools/`): Power grid utilities and analysis functions
- **Training** (`faraday/training/`): Data collection and model training capabilities
- **Test Networks** (`data/networks/`): Sample power grids including CIGRE test cases

## Installation

```bash
uv pip install -e .
```

## Usage

### CLI - Fix Network Violations

Run the automated workflow to fix violations in a power grid network:

```bash
# Fix violations in a network file
faraday network.json

# With verbose output and custom max iterations
faraday network.json --verbose --max-iterations 10
```

**CLI Options:**
- `--verbose, -v`: Enable detailed output including explanations
- `--max-iterations N`: Maximum number of fix iterations (default: 5)
- `--output, -o DIR`: Output directory for results (optional)

### Dashboard - Interactive Interface

Launch the Streamlit dashboard for interactive network analysis:

```bash
streamlit run faraday/dashboard.py
```

### Development

Run the LangGraph development server:

```bash
langgraph dev
```

## Features

- **Automated Violation Detection**: Identifies voltage and thermal violations
- **AI-Powered Planning**: Generates optimized action plans using LLMs
- **Smart Optimization**: Minimizes actions through coordinated interventions
- **Interactive Dashboard**: Web-based interface for network analysis
- **CLI Interface**: Command-line tool for batch processing
 