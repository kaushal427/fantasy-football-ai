# AI-Powered Fantasy Football Draft Assistant

An advanced fantasy football draft assistant that uses AI and machine learning to provide optimal draft recommendations based on real NFL data and 2025 rookie prospects.

## 🏗️ Project Structure

```
draft-optimizer/
├── core/                           # Core AI components
│   ├── __init__.py                # Package initialization
│   ├── ai_draft_assistant.py      # Main AI logic and ML models
│   └── news_analyzer.py           # News sentiment analysis
├── data/                          # Data storage (archive folder)
├── tests/                         # Test files
│   └── test_ai_assistant.py      # AI assistant tests
├── utils/                         # Utility functions
├── ai_app.py                      # Streamlit web application
├── main.py                        # Alternative entry point
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Option 1: Using the launcher (Recommended)
From the root directory:
```bash
python run_app.py
```

### Option 2: Direct execution
From the draft-optimizer directory:
```bash
cd draft-optimizer
streamlit run ai_app.py
```

### Option 3: Using main.py
From the draft-optimizer directory:
```bash
cd draft-optimizer
python main.py
```

## 📋 Features

### 🧠 AI-Powered Analysis
- **Machine Learning Models**: Predictive performance models for NFL veterans
- **Rookie Projections**: Advanced algorithms for 2025 draft prospects
- **Neural Network-Inspired Scoring**: Multi-layer scoring system
- **Real-time Adaptation**: Dynamic strategy based on draft position and available players

### 📊 Data Processing
- **9 CSV Files**: Comprehensive NFL data from the last 12 seasons
- **AI Preprocessing**: Outlier detection, missing value imputation, normalization
- **Feature Engineering**: Advanced predictive variables
- **Active Player Classification**: 2024/2023 veterans + 2025 rookies

### 🎯 Draft Strategy
- **Realistic ADP Integration**: Players recommended based on actual draft availability
- **Position Scarcity Analysis**: Intelligent roster construction
- **Value-Based Drafting**: Best player available with strategic considerations
- **Snake Draft Optimization**: Position-specific strategy for different draft spots

### 🖥️ User Interface
- **8 Comprehensive Tabs**: Complete draft analysis and strategy
- **Interactive Visualizations**: Charts and graphs for player analysis
- **Real-time Recommendations**: Live draft assistance
- **League Settings**: Customizable scoring and roster requirements

## 🔧 Installation

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   cd draft-optimizer
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python run_app.py
   ```

## 📁 Data Sources

The system processes 9 CSV files:
- `weekly_player_stats_offense.csv` / `weekly_player_stats_defense.csv`
- `yearly_player_stats_offense.csv` / `yearly_player_stats_defense.csv`
- `weekly_team_stats_offense.csv` / `weekly_team_stats_defense.csv`
- `yearly_team_stats_offense.csv` / `yearly_team_stats_defense.csv`
- `nfl_combine.csv` (2025 draft prospects)

## 🎮 League Settings

Default configuration:
- **Draft Type**: Snake draft
- **Teams**: 12
- **Scoring**: 0.5 PPR
- **Starting Positions**: 1 QB, 2 RB, 2 WR, 1 TE, 1 DEF, 1 K
- **Bench**: 6 players
- **IR**: 2 players

## 🧪 Testing

Run the test suite:
```bash
cd draft-optimizer
python -m pytest tests/
```

Or run individual tests:
```bash
cd draft-optimizer
python tests/test_ai_assistant.py
```

## 🔄 Recent Updates

- **Realistic ADP Integration**: Players now recommended based on actual draft availability
- **Improved Rookie Analysis**: Better handling of missing combine data
- **Enhanced Error Handling**: Robust column existence checks
- **Code Reorganization**: Clean, modular structure for better maintainability

## 📝 Usage

1. **Launch the application**
2. **Configure league settings** in the "📋 League Settings" tab
3. **Generate AI draft strategy** for your draft position
4. **Use live AI assistant** during your draft
5. **Analyze player rankings** and rookie prospects
6. **Review AI analytics** and model insights

## 🤝 Contributing

The code is organized into logical modules:
- `core/`: Core AI logic and ML models
- `tests/`: Test files for validation
- `data/`: Data storage and processing
- `utils/`: Utility functions and helpers

## 📞 Support

For issues or questions, please check the test files first to ensure the system is working correctly.
