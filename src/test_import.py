"""
File kiểm tra import các thư viện
"""

def test_imports():
    print("Kiểm tra import các thư viện cần thiết...")
    
    try:
        import pandas as pd
        print(f"✓ pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ pandas: {e}")
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
    
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
    
    try:
        import seaborn as sns
        print(f"✓ seaborn {sns.__version__}")
    except ImportError as e:
        print(f"✗ seaborn: {e}")
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
    
    try:
        import statsmodels
        print(f"✓ statsmodels {statsmodels.__version__}")
    except ImportError as e:
        print(f"✗ statsmodels: {e}")
    
    print("\nKiểm tra hoàn tất!")

if __name__ == "__main__":
    test_imports() 