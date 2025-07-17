import streamlit as st
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import streamlit as st



def quantile_slider(label, series, key, num_bins=10):
    try:
        # Clean numeric values
        series_numeric = pd.to_numeric(series, errors="coerce").dropna()
        if series_numeric.empty:
            st.warning(f"No numeric data available for '{label}'")
            return None

        # Convert to int for cleaner bin labels
        series_int = series_numeric.astype(int)

        # Compute quantile-based bin boundaries
        quantile_bins = np.unique(np.quantile(series_int, np.linspace(0, 1, num_bins + 1)).astype(int))

        if len(quantile_bins) < 2:
            st.warning(f"Not enough unique quantile bins for '{label}'")
            return None

        # Define session state key
        session_key = f"{key}_filter"
        default_range = (quantile_bins[0], quantile_bins[-1])

        # Initialize session state
        if session_key not in st.session_state:
            st.session_state[session_key] = default_range

        # Show select_slider with quantile bins
        selected_range = st.sidebar.select_slider(
            label,
            options=quantile_bins.tolist(),
            value=st.session_state[session_key]
        )

        # Return only if user narrowed the range
        if selected_range != default_range:
            return selected_range
        else:
            return None

    except Exception as e:
        st.warning(f"Error in quantile_slider for {label}: {e}")
        return None


def checkbox_filter(label_prefix, key_prefix):
    st.sidebar.markdown(f"**{label_prefix}**")

    with st.sidebar.container(): 
        cols = st.columns([1, 1])
        with cols[0]:
            f_checked = st.checkbox("female", key=f"{key_prefix}_f", label_visibility="visible")
        with cols[1]:
            m_checked = st.checkbox("male", key=f"{key_prefix}_m", label_visibility="visible")

    selected = []
    if f_checked:
        selected.append("f")
    if m_checked:
        selected.append("m")
    return selected



def get_filter_inputs(data):
    filters = {}
    st.sidebar.header("General Filters")

    if "filter_initial_data" not in st.session_state:
        st.session_state.filter_initial_data = {}

    disable = st.sidebar.checkbox(
        "Disable all general filters",
        value=st.session_state.get("filters-disabled", False),
        key="filters-disabled"
    )

    if disable:
        return {"__disable_all__": True}

    bd_rating_options = sorted(data["BD_rating"].dropna().unique())
    bd_key = "Bechdel Ratings"
    if bd_key in st.session_state:
        selected_bd_ratings = st.sidebar.multiselect("Bechdel Ratings", bd_rating_options, key=bd_key)
    else:
        selected_bd_ratings = st.sidebar.multiselect("Bechdel Ratings", bd_rating_options)
    if selected_bd_ratings:
        filters["BD_rating"] = selected_bd_ratings


    # Director Gender
    selected_genders = checkbox_filter("Director", "director_gender")
    if selected_genders:
        filters["director_gender"] = selected_genders

    #writer gender
    selected_genders = checkbox_filter("Writer", "writer_gender")
    if selected_genders:
        filters["writer_gender"] = selected_genders

    #actor gender
    actor1_genders = checkbox_filter("Primary Actor", "actor1_gender")
    if actor1_genders:
        filters["actor_1_gender"] = actor1_genders

    actor2_genders = checkbox_filter("Secondary", "actor2_gender")
    if actor2_genders:
        filters["actor_2_gender"] = actor2_genders

    # Avg vote
    min_vote, max_vote = float(data["avg_vote"].min()), float(data["avg_vote"].max())
    if "avg_vote" in st.session_state:
        avg_vote_range = st.sidebar.slider("Average Vote", min_vote, max_vote, key="avg_vote")
    else:
        avg_vote_range = st.sidebar.slider("Average Vote", min_vote, max_vote, (min_vote, max_vote), key="avg_vote")
    if avg_vote_range != (min_vote, max_vote):
        filters["avg_vote"] = avg_vote_range

    # Year
    min_year, max_year = int(data["year"].min()), int(data["year"].max())
    if "year_slider" in st.session_state:
        year_range = st.sidebar.slider("Release Year Range", min_year, max_year, key="year_slider")
    else:
        year_range = st.sidebar.slider("Release Year Range", min_year, max_year, (min_year, max_year), key="year_slider")
    if year_range != (min_year, max_year):
        filters["year"] = year_range

    # Genres
    genre_cols = ["genre_1", "genre_2", "genre_3"]
    genre_series = pd.Series(data[genre_cols].values.ravel()).dropna()
    genre_counts = genre_series.value_counts()
    genre_options = [f"{name} ({count})" for name, count in genre_counts.items()]
    genre_labels_key = "Genres"
    if genre_labels_key in st.session_state:
        selected_genre_labels = st.sidebar.multiselect("Genres", genre_options, key=genre_labels_key)
    else:
        selected_genre_labels = st.sidebar.multiselect("Genres", genre_options)
    selected_genres = [g.split(" (")[0] for g in selected_genre_labels]
    if selected_genres:
        filters["genres"] = selected_genres

    # Duration
    duration_range = quantile_slider("Duration (minutes)", data["duration"], key="duration")
    if duration_range:
        filters["duration"] = duration_range


    votes_range = quantile_slider("Votes", data["votes"], key="votes")
    if votes_range:
        filters["votes"] = votes_range

    budget_range = quantile_slider("Budget", data["budget"], key="budget", num_bins=20)
    if budget_range:
        filters["budget"] = budget_range

    usa_income_range = quantile_slider("USA Gross Income", data["usa_gross_income"], key="usa_gross", num_bins=20)
    if usa_income_range:
        filters["usa_gross_income"] = usa_income_range

    world_income_range = quantile_slider("Worldwide Gross Income", data["worlwide_gross_income"], key="world_gross", num_bins=20)
    if world_income_range:
        filters["worlwide_gross_income"] = world_income_range

    # Country
    country_counts = data["country"].dropna().value_counts()
    country_options = [f"{name} ({count})" for name, count in country_counts.items()]
    country_key = "Country"
    if country_key in st.session_state:
        selected_country_label = st.sidebar.selectbox("Country", ["All"] + country_options, key=country_key)
    else:
        selected_country_label = st.sidebar.selectbox("Country", ["All"] + country_options)
    selected_country = selected_country_label.split(" (")[0]
    if selected_country != "All":
        filters["country"] = selected_country

    # Language
    language_counts = data["language_1"].dropna().value_counts()
    language_options = [f"{name} ({count})" for name, count in language_counts.items()]
    language_key = "Primary Language"
    if language_key in st.session_state:
        selected_language_label = st.sidebar.selectbox("Primary Language", ["All"] + language_options, key=language_key)
    else:
        selected_language_label = st.sidebar.selectbox("Primary Language", ["All"] + language_options)
    selected_language = selected_language_label.split(" (")[0]
    if selected_language != "All":
        filters["language"] = selected_language

    # Director
    directors = pd.concat([data["director_1"], data["director_2"]]).dropna()
    director_counts = directors.value_counts()
    director_options = [f"{name} ({count})" for name, count in director_counts.items()]
    director_key = "Director"
    if director_key in st.session_state:
        selected_director_label = st.sidebar.selectbox("Director", ["All"] + director_options, key=director_key)
    else:
        selected_director_label = st.sidebar.selectbox("Director", ["All"] + director_options)
    selected_director = selected_director_label.split(" (")[0]
    if selected_director != "All":
        filters["director"] = selected_director

    # Writer
    writers = pd.concat([data["writer_1"], data["writer_2"]]).dropna()
    writer_counts = writers.value_counts()
    writer_options = [f"{name} ({count})" for name, count in writer_counts.items()]
    writer_key = "Writer"
    if writer_key in st.session_state:
        selected_writer_label = st.sidebar.selectbox("Writer", ["All"] + writer_options, key=writer_key)
    else:
        selected_writer_label = st.sidebar.selectbox("Writer", ["All"] + writer_options)
    selected_writer = selected_writer_label.split(" (")[0]
    if selected_writer != "All":
        filters["writer"] = selected_writer

    # Actor
    actors = pd.concat([data["actor_1"], data["actor_2"]]).dropna()
    actor_counts = actors.value_counts()
    actor_options = [f"{name} ({count})" for name, count in actor_counts.items()]
    actor_key = "Actor"
    if actor_key in st.session_state:
        selected_actor_label = st.sidebar.selectbox("Actor", ["All"] + actor_options, key=actor_key)
    else:
        selected_actor_label = st.sidebar.selectbox("Actor", ["All"] + actor_options)
    selected_actor = selected_actor_label.split(" (")[0]
    if selected_actor != "All":
        filters["actor"] = selected_actor

    return filters


debug = False

def apply_filters(data, filters):
    if filters.get("__disable_all__"):
        st.write("All filters disabled, returning full dataset.")
        return data, False

    filtered_data = data.copy()
    active = False

    def log_filter(name, before, after):
        if debug :
            st.write(f"Filter `{name}` reduced data from {before} to {after} rows.")

    before = len(filtered_data)

    if "year" in filters:
        y1, y2 = filters["year"]
        filtered_data = filtered_data[(filtered_data["year"] >= y1) & (filtered_data["year"] <= y2)]
        log_filter("year", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "genres" in filters:
        genre_cols = ["genre_1", "genre_2", "genre_3"]
        filtered_data = filtered_data[filtered_data[genre_cols].apply(
            lambda row: any(g in row.values for g in filters["genres"]), axis=1)]
        log_filter("genres", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "duration" in filters:
        d1, d2 = filters["duration"]
        filtered_data = filtered_data[(filtered_data["duration"] >= d1) & (filtered_data["duration"] <= d2)]
        log_filter("duration", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "country" in filters:
        filtered_data = filtered_data[filtered_data["country"] == filters["country"]]
        log_filter("country", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "language" in filters:
        filtered_data = filtered_data[filtered_data["language_1"] == filters["language"]]
        log_filter("language", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "director" in filters:
        d = filters["director"]
        filtered_data = filtered_data[(filtered_data["director_1"] == d) | (filtered_data["director_2"] == d)]
        log_filter("director", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "director_gender" in filters:
        filtered_data = filtered_data[filtered_data["director_1_gender"].isin(filters["director_gender"])]
        log_filter("director_gender", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "writer" in filters:
        w = filters["writer"]
        filtered_data = filtered_data[(filtered_data["writer_1"] == w) | (filtered_data["writer_2"] == w)]
        log_filter("writer", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "writer_gender" in filters:
        filtered_data = filtered_data[filtered_data["writer_1_gender"].isin(filters["writer_gender"])]
        log_filter("writer_gender", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "actor" in filters:
        a = filters["actor"]
        filtered_data = filtered_data[(filtered_data["actor_1"] == a) | (filtered_data["actor_2"] == a)]
        log_filter("actor", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "actor_1_gender" in filters:
        filtered_data = filtered_data[filtered_data["actor_1_gender"].isin(filters["actor_1_gender"])]
        log_filter("actor_1_gender", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "actor_2_gender" in filters:
        filtered_data = filtered_data[filtered_data["actor_2_gender"].isin(filters["actor_2_gender"])]
        log_filter("actor_2_gender", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "avg_vote" in filters:
        v1, v2 = filters["avg_vote"]
        filtered_data = filtered_data[(filtered_data["avg_vote"] >= v1) & (filtered_data["avg_vote"] <= v2)]
        log_filter("avg_vote", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "votes" in filters:
        v1, v2 = filters["votes"]
        filtered_data = filtered_data[(filtered_data["votes"] >= v1) & (filtered_data["votes"] <= v2)]
        log_filter("votes", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "budget" in filters:
        b1, b2 = filters["budget"]
        col = pd.to_numeric(filtered_data["budget"], errors="coerce")
        filtered_data = filtered_data[(col >= b1) & (col <= b2)]
        log_filter("budget", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "usa_gross_income" in filters:
        u1, u2 = filters["usa_gross_income"]
        col = pd.to_numeric(filtered_data["usa_gross_income"], errors="coerce")
        filtered_data = filtered_data[(col >= u1) & (col <= u2)]
        log_filter("usa_gross_income", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "worlwide_gross_income" in filters:
        w1, w2 = filters["worlwide_gross_income"]
        col = pd.to_numeric(filtered_data["worlwide_gross_income"], errors="coerce")
        filtered_data = filtered_data[(col >= w1) & (col <= w2)]
        log_filter("worlwide_gross_income", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    if "BD_rating" in filters:
        filtered_data = filtered_data[filtered_data["BD_rating"].isin(filters["BD_rating"])]
        log_filter("BD_rating", before, len(filtered_data))
        active = True
        before = len(filtered_data)

    return filtered_data, active

def filter_data(data, filters=None):
    if filters is None:
        filters = get_filter_inputs(data)
    return apply_filters(data, filters)


def apply_preset_filters(data):

    preset = st.session_state.pop("preset_filters", None)
    
    if preset:
        # Year range
        if "year" in preset:
            st.session_state.pop("year_slider", None)
            st.session_state["year_slider"] = preset["year"]

        # Genres
        if "genres" in preset:
            st.session_state.pop("Genres", None)
            st.session_state["Genres"] = preset["genres"]

        # Duration
        if "duration" in preset:
            st.session_state.pop("duration", None)
            st.session_state["duration"] = preset["duration"]

        # Country
        if "country" in preset:
            st.session_state.pop("Country", None)
            st.session_state["Country"] = preset["country"]

        # Language
        if "language" in preset:
            st.session_state.pop("Primary Language", None)
            st.session_state["Primary Language"] = preset["language"]

        # Director
        if "director" in preset:
            st.session_state.pop("Director", None)
            st.session_state["Director"] = preset["director"]

        # Gender filters 
        if "writer_gender" in preset:
            st.session_state["writer_gender_f"] = "f" in preset["writer_gender"]
            st.session_state["writer_gender_m"] = "m" in preset["writer_gender"]
        if "director_gender" in preset:
            st.session_state["director_gender_f"] = "f" in preset["director_gender"]
            st.session_state["director_gender_m"] = "m" in preset["director_gender"]
        if "actor_1_gender" in preset:
            st.session_state["actor1_gender_f"] = "f" in preset["actor_1_gender"]
            st.session_state["actor1_gender_m"] = "m" in preset["actor_1_gender"]
        if "actor_2_gender" in preset:
            st.session_state["actor2_gender_f"] = "f" in preset["actor_2_gender"]
            st.session_state["actor2_gender_m"] = "m" in preset["actor_2_gender"]

        # Votes, budget, etc.
        for field, key in [
            ("votes", "votes"),
            ("budget", "budget"),
            ("usa_gross_income", "usa_gross"),
            ("worlwide_gross_income", "world_gross"),
            ("avg_vote", "avg_vote"),
        ]:
            if field in preset:
                st.session_state.pop(key, None)
                st.session_state[key] = preset[field]

        # Bechdel ratings
        if "BD_rating" in preset:
            st.session_state["Bechdel Ratings"] = preset["BD_rating"]

    return filter_data(data)


