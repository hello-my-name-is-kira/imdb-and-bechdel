import os
import pandas as pd
import requests

from_file = 0
if from_file:
    script_dir = os.path.dirname(__file__)
    input_names_path = os.path.join(script_dir, "userdata", "names_to_lookup.csv")
    lookup_path = os.path.join(script_dir, "userdata", "full_name_gender_lookup.csv")
    output_names_path = os.path.join(script_dir, "userdata", "names_to_lookup.csv")

    names_df = pd.read_csv(input_names_path)
    all_names = names_df['full_name'].dropna().astype(str).str.strip()

    lookup_df = pd.read_csv(lookup_path)
    lookup_df['full_name_normalized'] = lookup_df['full_name'].astype(str).str.strip().str.lower()
    lookup_df = lookup_df.drop_duplicates(subset='full_name_normalized', keep='first')
    lookup_map = lookup_df.set_index('full_name_normalized')[['gender', 'certainty']].to_dict('index')

    output_rows = []
    missing_count = 0

    for name in all_names:
        norm = name.lower()
        match = lookup_map.get(norm, None)
        if match:
            gender = match['gender']
            certainty = match['certainty']
        else:
            gender = None
            certainty = None
            missing_count += 1
        output_rows.append({
            'full_name': name,
            'gender': gender,
            'certainty': certainty
        })

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_names_path, index=False)

    print(f"\n Saved {len(output_df)} full names with gender/certainty (where available) to:\n{os.path.abspath(output_names_path)}")
    if missing_count > 0:
        print(f" {missing_count} names are still missing gender/certainty.")
    else:
        print("All names are matched with gender and certainty.")

else:
    script_dir = os.path.dirname(__file__)
    input_names_path = os.path.join(script_dir, "userdata", "gender_match.csv")
    names_df = pd.read_csv(input_names_path)

    updated_rows = []
    missing_names = names_df[names_df['gender'].isna() | names_df['gender'].eq('')].copy()
    print("\n Updating gender info for missing names (will stop on API error)")

    for idx, row in missing_names.iterrows():
        name = row['full_name']
        first_name = name.split()[0].lower()
        try:
            r = requests.get(f"https://api.genderize.io?name={first_name}")
            if r.status_code == 200:
                data = r.json()
                gender = data.get('gender')
                probability = data.get('probability', 0.0)
                names_df.at[idx, 'gender'] = {'male': 'm', 'female': 'f'}.get(gender, 'u')
                names_df.at[idx, 'certainty'] = probability
                print(f"{name} → Gender: {gender} (prob: {probability})")
            else:
                print(f"{name} → API status {r.status_code}, stopping.")
                break
        except Exception as e:
            print(f"{name} → API error: {e}, stopping.")
            break

    names_df.to_csv(input_names_path, index=False)
    print(f"\nUpdated names with gender saved to: {os.path.abspath(input_names_path)}")

    still_missing = names_df[names_df['gender'].isna() | names_df['gender'].eq('')]
    print(f"\n {len(still_missing)} names are still missing gender/certainty.")