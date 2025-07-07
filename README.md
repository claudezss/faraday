# EnergiQ-Agent

AI-powered power grid optimization system that automatically detects and resolves electrical network violations.

## Installation

```bash
uv pip install -e .
```

## Usage

### CLI - Fix Network Violations

Run the automated workflow to fix violations in a power grid network:

```bash
# Fix violations in a network file
energiq-agent network.json

# With verbose output and custom max iterations
energiq-agent network.json --verbose --max-iterations 10
```

**CLI Options:**
- `--verbose, -v`: Enable detailed output including explanations
- `--max-iterations N`: Maximum number of fix iterations (default: 5)
- `--output, -o DIR`: Output directory for results (optional)

### Dashboard - Interactive Interface

Launch the Streamlit dashboard for interactive network analysis:

```bash
streamlit run energiq_agent/dashboard.py
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
 