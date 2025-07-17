import os
import pandas as pd

script_dir = os.path.dirname(__file__)
input_path = os.path.join(script_dir, "userdata", "imdb00-20merged.csv")
output_path = os.path.join(script_dir, "userdata", "imdb00-20merged-cleaned.csv")

df = pd.read_csv(input_path)

df_cleaned = df.dropna(subset=['BD_rating']).copy()

def extract_actor_2(actor_field):
    if pd.isna(actor_field):
        return actor_field
    parts = actor_field.split(',')
    return parts[1].strip() if len(parts) > 1 else None

df_cleaned['actors_f2'] = df_cleaned['actors_f2'].apply(extract_actor_2)

genre_split = df_cleaned['genre'].str.split(',', expand=True)
genre_split.columns = ['genre_1', 'genre_2', 'genre_3']
for col in genre_split.columns:
    genre_split[col] = genre_split[col].map(lambda x: x.strip() if isinstance(x, str) else x)

genre_index = df_cleaned.columns.get_loc('genre')
for i, col in enumerate(genre_split.columns):
    df_cleaned.insert(genre_index + 1 + i, col, genre_split[col])

writer_split = df_cleaned['writer'].str.split(',', expand=True)
writer_split.columns = ['writer_1', 'writer_2']
for col in writer_split.columns:
    writer_split[col] = writer_split[col].map(lambda x: x.strip() if isinstance(x, str) else x)

writer_index = df_cleaned.columns.get_loc('writer')
df_cleaned.insert(writer_index + 1, 'writer_1', writer_split['writer_1'])
df_cleaned.insert(writer_index + 2, 'writer_2', writer_split['writer_2'])


df_cleaned.rename(columns={
    'actors_1': 'actor_1',
    'actors_f2': 'actor_2'
}, inplace=True)


director_split = df_cleaned['director'].str.split(',', expand=True)
director_split.columns = ['director_1', 'director_2']
for col in director_split.columns:
    director_split[col] = director_split[col].map(lambda x: x.strip() if isinstance(x, str) else x)

director_index = df_cleaned.columns.get_loc('director')
df_cleaned.insert(director_index + 1, 'director_1', director_split['director_1'])
df_cleaned.insert(director_index + 2, 'director_2', director_split['director_2'])

df_cleaned.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to:\n{os.path.abspath(output_path)}")
print(f"Remaining rows after dropping: {len(df_cleaned)}")

columns_to_check = ['actor_1', 'actor_2', 'writer_1', 'writer_2', 'director_1', 'director_2']


