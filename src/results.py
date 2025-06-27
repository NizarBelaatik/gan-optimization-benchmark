from pytorch_fid import fid_score
import os
from config import config
import matplotlib.pyplot as plt
import pandas as pd
import os

sample_dir = config.dirs['samples']

print("\nContents of outputs/samples:")
if os.path.exists(sample_dir):
    for fname in os.listdir(sample_dir):
        print(" -", fname)
else:
    print("❌ Directory not found:", sample_dir)
