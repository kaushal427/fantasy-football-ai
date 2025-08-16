# Data Directory Structure

- `/processed/`: Cleaned and processed data files (<50MB)
- `/sample/`: Sample data files for testing
- `/archive/`: Large raw data files (not tracked in Git)

## Large Data Files
Large data files are not tracked in Git. To get the complete dataset:
1. Download from [data source URL]
2. Place files in the `/archive/` directory
3. Run data processing scripts