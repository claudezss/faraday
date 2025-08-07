# Faraday Dashboard v2.0

Advanced Streamlit dashboard for AI-powered power grid violation analysis and resolution.

## 🚀 Features

### 🎛️ Multiple Operation Modes
- **Auto Mode**: Fully automated violation resolution without user intervention
- **Interactive Mode**: Step-by-step workflow with plan approval and modification
- **Expert Mode**: Advanced manual controls with detailed parameter tuning

### 📊 Advanced Visualization
- **Interactive Network Plots**: Pandapower integration with zoom, pan, and hover
- **Violation Heatmaps**: Color-coded severity visualization
- **Before/After Comparison**: Side-by-side network state analysis
- **Real-time Updates**: Live network status during workflow execution

### ⚡ Smart Action Planning
- **Visual Action Editor**: Drag-and-drop action sequencing
- **Parameter Optimization**: Intelligent parameter suggestions
- **Impact Preview**: Estimated effectiveness visualization
- **Plan Validation**: Real-time feasibility checking

### 📈 Comprehensive Analysis
- **Detailed Metrics**: Network health, power flow, and violation statistics
- **Effectiveness Tracking**: Action impact analysis and success rates
- **Historical Trends**: Performance tracking over multiple runs
- **Export & Reporting**: PDF reports and CSV data export

## 🏗️ Architecture

```
faraday/dashboard/
├── main.py                    # Main dashboard application
├── components/               # Modular UI components
│   ├── network_viz.py       # Network visualization
│   ├── action_editor.py     # Action plan editor
│   ├── status_panel.py      # Status monitoring
│   └── comparison_view.py   # Before/after comparison
├── utils/                   # Utilities
│   ├── session_state.py     # Session management
│   └── data_processing.py   # Data analysis
└── assets/                  # Static files
    └── styles.css          # Custom styling
```

## 🚦 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements-dashboard.txt
```

### 2. Launch Dashboard
```bash
# Option 1: Direct launch
python run_dashboard.py

# Option 2: Streamlit command
streamlit run faraday/dashboard/main.py

# Option 3: Python module
python -m faraday.dashboard.main
```

### 3. Access Dashboard
Open your browser to [http://localhost:8501](http://localhost:8501)

## 📋 Usage Guide

### Getting Started
1. **Upload Network**: Use the sidebar to upload a pandapower JSON file
2. **Configure Settings**: Adjust voltage thresholds and other parameters
3. **Select Mode**: Choose Auto, Interactive, or Expert mode
4. **Analyze & Fix**: Let the AI analyze and resolve violations

### Auto Mode
- Click "🚀 Start Automatic Resolution"
- Watch real-time progress
- Review results and export reports

### Interactive Mode
- Start interactive session
- Review AI-generated action plans
- Approve, modify, or reject plans
- Execute approved actions step-by-step

### Expert Mode
- Manual action plan creation
- Advanced parameter tuning
- Custom violation analysis
- Detailed debugging tools

## 🎨 User Interface

### Header & Navigation
- **Main Header**: Project branding and current status
- **Mode Selection**: Easy switching between operation modes
- **Progress Tracking**: Real-time workflow progress

### Sidebar Configuration
- **File Upload**: Network file management
- **Voltage Thresholds**: Configurable violation limits
- **Session Info**: Current session statistics
- **Export Options**: Report and data export

### Main Content Area
- **Network Visualization**: Interactive plots and analysis
- **Action Plan Editor**: Visual plan building and modification
- **Status Dashboard**: Real-time monitoring and alerts
- **Results Comparison**: Before/after analysis

## 🔧 Configuration

### Environment Variables
```bash
# Streamlit configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_SERVER_HEADLESS=true

# Faraday configuration
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
```

### Voltage Thresholds
- **Default**: v_min=0.95, v_max=1.05 p.u.
- **Configurable**: Via sidebar controls
- **Persistent**: Settings saved per session

## 📊 Data Flow

1. **Network Upload**: User uploads pandapower JSON file
2. **Analysis**: AI analyzes network status and violations
3. **Plan Generation**: AI creates optimized action plan
4. **User Interaction**: Review, modify, or approve plan
5. **Execution**: Actions executed on network copy
6. **Validation**: Results validated and compared
7. **Reporting**: Generate reports and export data

## 🔌 Integration

### Faraday Core
- **Workflow Integration**: Uses faraday.agents.workflow
- **Tool Integration**: Leverages faraday.tools.pandapower
- **State Management**: Compatible with faraday.agents.workflow.state

### External Libraries
- **Streamlit**: Web interface framework
- **Plotly**: Interactive visualization
- **Pandas**: Data manipulation
- **Pandapower**: Power system analysis

## 🛠️ Development

### Adding New Components
1. Create component in `components/` directory
2. Follow existing component structure
3. Update `__init__.py` imports
4. Add to main dashboard layout

### Custom Styling
- Modify `assets/styles.css`
- Use CSS variables for consistency
- Follow responsive design principles

### Session State
- Use `SessionStateManager` for all state operations
- Maintain state consistency across components
- Handle state persistence and recovery

## 🚨 Troubleshooting

### Common Issues

**Dashboard won't start:**
- Check Python version (>=3.8)
- Install required dependencies
- Verify network file permissions

**Network visualization errors:**
- Ensure pandapower is installed
- Check network file format
- Verify power flow convergence

**Performance issues:**
- Use appropriate network size limits
- Enable compression for large networks
- Monitor memory usage

### Debug Mode
Enable debug information:
```python
# In sidebar
st.checkbox("🐛 Debug Mode")
```

## 📈 Performance

### Optimization Features
- **Lazy Loading**: Components loaded on demand
- **Data Caching**: Streamlit caching for network analysis
- **Progressive Rendering**: Large networks rendered progressively
- **Memory Management**: Automatic cleanup of temporary data

### Scale Limits
- **Small Networks**: <100 buses (full visualization)
- **Medium Networks**: 100-500 buses (hierarchical view)
- **Large Networks**: 500+ buses (compressed representation)

## 🔐 Security

### Data Privacy
- **Local Processing**: All data processed locally
- **No External Uploads**: Network data stays on your machine
- **Session Isolation**: Each session is independent

### API Security
- **Environment Variables**: API keys stored securely
- **Request Validation**: All inputs validated
- **Error Handling**: Secure error messages

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Follow coding standards
4. Add tests for new features
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Pandapower**: Power system analysis library
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **LangGraph**: AI workflow orchestration