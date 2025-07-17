


import os
import pandas as pd

script_dir = os.path.dirname(__file__)
input_path = os.path.join(script_dir, "userdata", "imdb00-20merged-cleaned.csv")
output_names_path = os.path.join(script_dir, "userdata", "names_to_lookup.csv")

df = pd.read_csv(input_path)

columns = ['writer_1', 'writer_2', 'actor_1', 'actor_2', 'director']

print("\n🔎 Names from the first 3 entries in the dataset:")
for i in range(min(3, len(df))):
    print(f"\n🎬 Entry {i+1}:")
    for col in columns:
        name = df.at[i, col] if pd.notna(df.at[i, col]) else "<missing>"
        print(f"  {col}: {name}")

all_names = set()
for col in columns:
    if col in df.columns:
        names = df[col].dropna().astype(str).str.strip()
        all_names.update(names)

names_to_lookup = sorted(all_names)

pd.DataFrame({'full_name': names_to_lookup}).to_csv(output_names_path, index=False)

print(f"\n Saved {len(names_to_lookup)} full names in alphabetical order to:\n{os.path.abspath(output_names_path)}")
