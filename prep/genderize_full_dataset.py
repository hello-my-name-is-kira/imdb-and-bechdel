import os
import pandas as pd

script_dir = os.path.dirname(__file__)
input_path = os.path.join(script_dir, "userdata", "imdb00-20merged-cleaned.csv")
gender_path = os.path.join(script_dir, "userdata", "gender_match.csv")
output_path = os.path.join(script_dir, "userdata", "imdb00-20merged-cleaned-gendered.csv")

gender_df = pd.read_csv(gender_path)
gender_lookup = gender_df.set_index('full_name')['gender'].to_dict()

name_columns = ['actor_1', 'actor_2', 'writer_1', 'writer_2', 'director_1', 'director_2']
drop_columns = ['genre', 'director', 'writer', 'imdb_id_clean', 'BD_title']
chunk_size = 1000
header_written = False

if os.path.exists(output_path):
    os.remove(output_path)
    print(f"🧹 Removed previous output file: {output_path}")

reader = pd.read_csv(input_path, chunksize=chunk_size)
chunk_num = 1

for chunk in reader:
    original_rows = len(chunk)
    print(f"\n Processing chunk {chunk_num} with {original_rows} rows...")

    for col in name_columns:
        before = len(chunk)
        chunk = chunk[~chunk[col].astype(str).str.contains(',')]
        removed = before - len(chunk)
        if removed > 0:
            print(f"    Removed {removed} rows from column '{col}' (comma found)")

    for col in drop_columns:
        if col in chunk.columns:
            chunk.drop(columns=col, inplace=True)
            print(f"    Dropped column: {col}")

    for col in reversed(name_columns): 
        gender_col = f"{col}_gender"
        gender_data = chunk[col].map(gender_lookup)
        insert_pos = chunk.columns.get_loc(col) + 1
        chunk.insert(insert_pos, gender_col, gender_data)
        print(f"   Added gender column: {gender_col}")

    chunk.to_csv(output_path, mode='a', index=False, header=not header_written)
    print(f"   Written {len(chunk)} rows to file.")

    header_written = True
    chunk_num += 1

print(f"\n Gender-enhanced dataset saved to:\n{os.path.abspath(output_path)}")
