# split_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

DATA_PATH = Path("data/raw")
OUTPUT_JSON = Path("data/splits")

def split_section(machine, section):
    attr = pd.read_csv(DATA_PATH / machine / f"attributes_{section}.csv")
    
    # 1. Source train normal → train/val
    src_train = attr[attr['file_name'].str.contains('source_train_normal')]
    train, val = train_test_split(src_train, test_size=0.2, random_state=42, stratify=src_train['d1v'])
    
    # 2. Test source + target
    test_source = attr[attr['file_name'].str.contains('source_test_')]
    test_target = attr[attr['file_name'].str.contains('target_test_')]
    
    # Lưu đường dẫn
    def save_paths(df, name):
        paths = (DATA_PATH / df['file_name']).astype(str).tolist()
        return {f"{machine}_{section}_{name}": paths}
    
    return {
        **save_paths(train, "train"),
        **save_paths(val, "val"),
        **save_paths(test_source, "test_source"),
        **save_paths(test_target, "test_target")
    }

# Chạy cho tất cả
splits = {}
for machine in ['fan', 'valve']:
    for sec in ['00', '01', '02']:
        splits.update(split_section(machine, sec))

# Lưu JSON
OUTPUT_JSON.mkdir(exist_ok=True)
with open(OUTPUT_JSON / "splits.json", "w") as f:
    json.dump(splits, f, indent=2)