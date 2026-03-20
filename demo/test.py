from data_load import load_dataset
from preprocessing import inspect_dataset, preprocess_dataset
from pathlib import Path

csv_path = Path(__file__).resolve().parent / "data" / "adhdata.csv"
df = load_dataset(csv_path)
inspect_dataset(df)

print("Dataset preprocesado")
df_clean, eeg_cols = preprocess_dataset(df)
inspect_dataset(df_clean)
