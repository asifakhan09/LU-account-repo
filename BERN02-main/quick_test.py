import numpy as np
import pandas as pd
print("Success! All imports working")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Quick test
data = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame({'numbers': data})
print("Basic operations working")
print(df)