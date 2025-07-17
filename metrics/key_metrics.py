import streamlit as st
import pandas as pd

def display_gender_composition_metrics(data):
    gender_cols = [
        "director_1_gender", "director_2_gender",
        "writer_1_gender", "writer_2_gender",
        "actor_1_gender", "actor_2_gender"
    ]

    def valid_and_consistent(row, gender):
        known_genders = [g for g in row if pd.notna(g)]
        return len(known_genders) >= 3 and all(g == gender for g in known_genders)

    filtered = data[gender_cols]
    total_valid = filtered.apply(lambda row: len([g for g in row if pd.notna(g)]) >= 3, axis=1).sum()

    all_male_pct = (filtered.apply(lambda row: valid_and_consistent(row, "m"), axis=1).sum() / total_valid) * 100 if total_valid > 0 else 0
    all_female_pct = (filtered.apply(lambda row: valid_and_consistent(row, "f"), axis=1).sum() / total_valid) * 100 if total_valid > 0 else 0

    def percent_female(col):
        series = data[col].dropna().str.lower().str.strip()
        return (series == "f").mean() * 100 if len(series) > 0 else 0

    pct_female_directors = percent_female("director_1_gender")
    pct_female_writers = percent_female("writer_1_gender")
    pct_female_actors = percent_female("actor_1_gender")

    pct_bechdel_3 = (data["BD_rating"] == 3).mean() * 100 if "BD_rating" in data.columns else 0

    baseline = st.session_state.get("baseline_metrics", {})

    metrics = [
        ("All-Male Key Roles", all_male_pct, baseline.get("all_male", None), "This is the percentage of movies where all key roles (directors, writers, primary and secondary actor) are men"),
        ("All-Female Key Roles", all_female_pct, baseline.get("all_female", None), "This is the percentage of movies where all key roles (directors, writers, primary and secondary actor) are women"),
        ("Female Directors", pct_female_directors, baseline.get("female_director", None), None),
        ("Female Writers", pct_female_writers, baseline.get("female_writer", None), None),
        ("Female Lead Actors", pct_female_actors, baseline.get("female_actor", None), None),
        ("Bechdeltest passed", pct_bechdel_3, baseline.get("bechdel_3", None), None)
    ]

    cols = st.columns(len(metrics))
    for col, (label, value, base, help) in zip(cols, metrics):
        delta = None if base is None else round(value - base, 1)
        with col:
            st.metric(label, f"{value:.1f}%", delta=f"{delta:+.1f}%" if delta is not None else None, help=help if help is not None else None)
