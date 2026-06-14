import sys, os, time

# Import pandas and numpy FIRST
print("0. Importing pandas and numpy...", flush=True)
t0 = time.time()
import numpy as np
import pandas as pd
print(f"   pandas and numpy imported in {time.time()-t0:.2f}s", flush=True)

sys.path.insert(0, 'src')

print("1. Importing evaluate...", flush=True)
t0 = time.time()
from evaluate import compute_eer
print(f"   evaluate imported in {time.time()-t0:.2f}s", flush=True)

print("2. Importing models...", flush=True)
t0 = time.time()
from models import get_model
print(f"   models imported in {time.time()-t0:.2f}s", flush=True)

print("3. Importing dataset...", flush=True)
t0 = time.time()
import dataset
print(f"   dataset imported in {time.time()-t0:.2f}s", flush=True)

print("4. Starting granular steps of build_file_index...", flush=True)
data_root = "D:\\kaggle-data\\for-norm"
split = "training"
records = []
split_dir = os.path.join(data_root, 'for-norm', split)
print(f"   split_dir: {split_dir}", flush=True)

for label_name, label_id in [('real', 0), ('fake', 1)]:
    class_dir = os.path.join(split_dir, label_name)
    print(f"   Listing {class_dir}...", flush=True)
    t_start = time.time()
    files = os.listdir(class_dir)
    print(f"   Done listing {class_dir} in {time.time()-t_start:.2f}s, found {len(files)} entries", flush=True)
    
    print(f"   Filtering and appending dictionary records for {label_name}...", flush=True)
    t_start = time.time()
    for fname in files:
        if fname.lower().endswith('.wav'):
            records.append({
                'path': os.path.join(class_dir, fname),
                'label': label_id,
                'split': split,
                'class_name': label_name
            })
    print(f"   Done filtering for {label_name} in {time.time()-t_start:.2f}s, records list size is now {len(records)}", flush=True)

print("5. Converting records to pandas DataFrame...", flush=True)
t_start = time.time()
df = pd.DataFrame(records)
print(f"   Done converting in {time.time()-t_start:.2f}s, shape: {df.shape}", flush=True)

print("6. Printing summary...", flush=True)
print(f"[{split}] Found {len(df)} files - Real: {(df['label'] == 0).sum()}, Fake: {(df['label'] == 1).sum()}", flush=True)
print("ALL STEPS PASSED SUCCESSFULLY!", flush=True)
