print('='*60)
print('TEST MAIN.PY - STARTING')
print('='*60)

import sys
print(f'Python path: {sys.path}')

try:
    from src.utils import ensure_directories, setup_logging, load_config
    print('✅ utils.py imported')
except Exception as e:
    print(f'❌ utils.py import failed: {e}')

try:
    from src.data_preprocessing import DataPreprocessor
    print('✅ data_preprocessing.py imported')
except Exception as e:
    print(f'❌ data_preprocessing.py import failed: {e}')

try:
    from src.feature_engineering import FeatureEngineer
    print('✅ feature_engineering.py imported')
except Exception as e:
    print(f'❌ feature_engineering.py import failed: {e}')

print('='*60)
print('TEST COMPLETE')
print('='*60)
