import os
import pandas as pd

script_dir = os.path.dirname(__file__)
input_path = os.path.join(script_dir, "userdata", "imdb00-20merged-cleaned.csv")
output_path = os.path.join(script_dir, "userdata", "gender_match.csv")

df = pd.read_csv(input_path)

name_columns = ['actor_1', 'actor_2', 'writer_1', 'writer_2', 'director_1', 'director_2']

all_names = set()
for col in name_columns:
    if col in df.columns:
        names = df[col].dropna().astype(str).str.strip()
        all_names.update(names)

gender_df = pd.DataFrame({'full_name': sorted(all_names)})
gender_df['gender'] = None
gender_df['certainty'] = None

gender_df.to_csv(output_path, index=False)
print(f" Saved {len(gender_df)} unique names to:\n{os.path.abspath(output_path)}")



script_dir = os.path.dirname(__file__)
match_path = os.path.join(script_dir, "userdata", "gender_match.csv")
lookup_path = os.path.join(script_dir, "userdata", "names_to_lookup.csv")

gender_match = pd.read_csv(match_path)
lookup = pd.read_csv(lookup_path)

lookup_clean = lookup.dropna(subset=['gender', 'certainty'])
lookup_clean = lookup_clean[['full_name', 'gender', 'certainty']].copy()

merged = gender_match.merge(lookup_clean, on='full_name', how='left', suffixes=('', '_new'))

merged['gender'] = merged['gender_new'].combine_first(merged['gender'])
merged['certainty'] = merged['certainty_new'].combine_first(merged['certainty'])

merged = merged.drop(columns=['gender_new', 'certainty_new'])

merged.to_csv(match_path, index=False)
print(f" Gender and certainty merged into:\n{os.path.abspath(match_path)}")
print(f" Updated {lookup_clean.shape[0]} names with exact matches.")
