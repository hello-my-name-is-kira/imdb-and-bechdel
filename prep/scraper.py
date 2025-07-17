import os
import pandas as pd
import requests

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, "userdata", "IMDB Movies 2000 - 2020.csv")
imdb_df = pd.read_csv(csv_path)

imdb_df['imdb_id_clean'] = imdb_df['imdb_title_id'].str.replace("tt", "").str.zfill(7)

response = requests.get("https://bechdeltest.com/api/v1/getAllMovies")
bechdel_data = response.json()
bechdel_df = pd.DataFrame(bechdel_data)

bechdel_df['imdb_id_clean'] = bechdel_df['imdbid'].astype(str).str.zfill(7)

bechdel_df_prefixed = bechdel_df.rename(columns=lambda col: f"BD_{col}" if col != 'imdb_id_clean' else col)

merged_df = pd.merge(imdb_df, bechdel_df_prefixed, on='imdb_id_clean', how='left')

output_path = os.path.join(script_dir, "userdata", "imdb00-20merged.csv")
merged_df.to_csv(output_path, index=False)

print(f"Merged dataset saved to:\n{os.path.abspath(output_path)}")
