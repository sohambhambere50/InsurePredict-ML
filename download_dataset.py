import pandas as pd
import os 

# Create data directory
os.makedirs("data", exist_ok = True)

# Download from the reference repo
print("Downloading dataset from reference repo...")

try:
    # Train Data
    train_url = "https://raw.githubusercontent.com/prsdm/mlops-project/main/data/train.csv" 
    train_df = pd.read_csv(train_url)
    train_df.to_csv("data/train.csv", index=False)
    print(f"✅ Train data: {train_df.shape}")

    # Test data
    test_url = "https://raw.githubusercontent.com/prsdm/mlops-project/main/data/test.csv"
    test_df = pd.read_csv(test_url)
    test_df.to_csv("data/test.csv", index=False)
    print(f"✅ Test Data: {test_df.shape}")

    print(f"\n📊 Columns: {list(train_df.columns)}")
    print(f"\n🔍 First 3 rows:")
    print(train_df.head(3))

except Exception as e:
    print(f"❌ Error: {e}")