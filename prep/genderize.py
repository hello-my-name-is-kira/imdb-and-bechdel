import os
import pandas as pd
import requests
import time

MAX_API_CALLS = -1 

script_dir = os.path.dirname(__file__)
input_path = os.path.join(script_dir, "userdata", "imdb00-20merged-cleaned.csv")
output_lookup_path = os.path.join(script_dir, "userdata", "full_name_gender_lookup.csv")

df = pd.read_csv(input_path)

columns = ['writer_1', 'writer_2', 'actor_1', 'actor_2', 'director']

full_names = set()

for col in columns:
    names = df[col].dropna().astype(str).str.strip()
    full_names.update(names)

if os.path.exists(output_lookup_path):
    existing_lookup = pd.read_csv(output_lookup_path)
    known_names = set(existing_lookup['full_name'].dropna().astype(str).str.strip().str.lower())
else:
    existing_lookup = pd.DataFrame(columns=['full_name', 'gender', 'certainty'])
    known_names = set()


names_to_lookup = list(full_names - known_names)
print(f"🔍 {len(names_to_lookup)} new full names to process.")

results = []
api_calls = 0

for name in names_to_lookup:
    if MAX_API_CALLS != -1 and api_calls >= MAX_API_CALLS:
        print("Reached API call limit.")
        break

    try:
        first = name.split()[0].lower()
        r = requests.get(f"https://api.genderize.io?name={first}")
        if r.status_code != 200:
            print(f"API returned status {r.status_code}. Stopping.")
            break

        data = r.json()
        gender = data.get("gender")
        probability = data.get("probability", 0.0)

        gender_code = {'male': 'm', 'female': 'f'}.get(gender, 'u')
        results.append({
            "full_name": name,
            "gender": gender_code,
            "certainty": probability
        })

        api_calls += 1
        #time.sleep(0.0)

    except Exception as e:
        print(f"Error for name '{name}': {e}")
        break

if results:
    new_df = pd.DataFrame(results)
    final_df = pd.concat([existing_lookup, new_df], ignore_index=True)
    final_df.to_csv(output_lookup_path, index=False)
    print(f"\n️️ Saved gender lookup file with {len(final_df)} entries to:\n{os.path.abspath(output_lookup_path)}")
else:
    print("No new results saved.")
