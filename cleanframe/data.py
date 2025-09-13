import pandas as pd
import os

def sample_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_dir, "sample_data", "retail_store_sales.csv")

    return pd.read_csv(file_path)