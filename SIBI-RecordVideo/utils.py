"""
Helper untuk manajemen file & metadata.
"""
import os
import pandas as pd
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_next_filename(folder, base_name, ext="mp4"):
    """
    Auto-numbering: base_name_01.ext, base_name_02.ext, ...
    """
    existing = [f for f in os.listdir(folder) if f.startswith(base_name) and f.endswith(ext)]
    nums = []
    for f in existing:
        try:
            num = int(f.replace(base_name + "_", "").replace(f".{ext}", ""))
            nums.append(num)
        except:
            pass
    next_num = max(nums) + 1 if nums else 1
    return f"{base_name}_{next_num:02d}.{ext}"

def append_metadata(filename, label, duration, notes="", dataset_dir="dataset"):
    """
    Tambahkan baris baru ke metadata.csv di directory yang dipilih
    """
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "metadata.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_row = {
        "filename": filename,
        "label": label,
        "timestamp": timestamp,
        "duration": round(duration, 2),
        "notes": notes
    }
    df_new = pd.DataFrame([new_row])
    if os.path.isfile(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)