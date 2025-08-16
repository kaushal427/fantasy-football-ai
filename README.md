# Data Directory Structure

- `/processed/`: Cleaned and processed data files (<50MB)
- `/sample/`: Sample data files for testing
- `/archive/`: Large raw data files (not tracked in Git)

## Large Data Files
Large data files are not tracked in Git. To get the complete dataset:
1. Download from [https://www.kaggle.com/datasets/philiphyde1/nfl-stats-1999-2022?resource=download&select=weekly_player_stats_defense.csv]
2. Place files in the `/archive/` directory
3. Run data processing scripts
