import streamlit as st
st.set_page_config(layout="wide", page_title="IMDB and Bechdel", initial_sidebar_state='collapsed')
import pandas as pd
import os


from filters import apply_preset_filters
from metrics.key_metrics import display_gender_composition_metrics



def show_dashboard_warning():
    if "acknowledged_warning" not in st.session_state:
        st.session_state["acknowledged_warning"] = False

    if not st.session_state["acknowledged_warning"]:
        st.markdown(
            """
                <strong>⚠️ Warning:</strong><br>
                Do not use this dashboard as a source for anything. The data aggregation and data collection are 
                <strong>not representative</strong> for any form of educational or statistical work.<br>
                This is a university project that focuses on visualization, <strong>not on correctness of data</strong>.
            """,
            unsafe_allow_html=True
        )
        if st.button("✅ I understand"):
            st.session_state["acknowledged_warning"] = True
            st.rerun()
        st.stop()

show_dashboard_warning()

from plots import (
    bechdel_rating_per_year,
    avg_bechdel_score_by_genre_and_year,
    female_roles_over_time,
    bechdel_rating_by_gender_all_roles,
    avg_bechdel_score_by_genre_and_gendered_role,
    female_roles_by_genre_heatmap,
    scatter_budget_vs_income,
    income_and_budget_by_bechdel_pass,
    plot_bechdel_vs_rating,
    compare_financials_by_role_gender 
)

from plot_EDA import plot_selected_distribution, plot_financial_distribution, plot_categorical_donut

from plot_DT import sunburst, all_features

from plot_wordcloud import wordcloud, top_words_bar_chart, bar_race_by_year



file_path = "userdata/imdb00-20merged-cleaned-gendered.csv"


if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")

data = pd.read_csv(file_path)

# Combine genres as before
genre_cols = ["genre_1", "genre_2", "genre_3"]
data["all_genres"] = data[genre_cols].astype(str).apply(lambda x: [g for g in x if g and g != 'nan'], axis=1)

top_sidebar = st.sidebar.container()
with top_sidebar :
    st.empty()

    

filtered_data, _ = apply_preset_filters(data)


col_title, col_counter, col_exp = st.columns([8, 3, 6])
with col_title :
    st.title("IMDB and Bechdel")
with col_counter : 
    st.metric(
        label="Movies in current selection",
        value=f"{len(filtered_data):,}",
        delta=f"{len(filtered_data) - len(data)}",
        help="Every visualization in this dashboard can be influenced by the global filters in the sidebar (initially hidden)."
    )
with col_exp :
    st.markdown(
        """
        <div style='color:#b5b5b5'>
            1. The movie has to have at least two named women in it<br>
            2. Who talk to each other<br>
            3. About something besides a man
        </div>
        """,
        unsafe_allow_html=True
    )


pages = ["Analysis", "Bechdel", "Industry", "Tree", "Wordcloud" ,"Dataset"]

page = st.segmented_control(
    "",
    options=pages,
    default="Analysis",    
    key="page_selector"         
)







def init_baseline_director_rating_metrics(data):
    if "baseline_director_metrics" not in st.session_state:
        df = data.copy()
        df["director_1_gender"] = df["director_1_gender"].str.lower()
        df = df.dropna(subset=["director_1_gender", "avg_vote"])

        # Filter to male/female only
        df = df[df["director_1_gender"].isin(["m", "f"])]

        # Compute baseline averages
        baseline_metrics = {
            "female": df[df["director_1_gender"] == "f"]["avg_vote"].mean(),
            "male": df[df["director_1_gender"] == "m"]["avg_vote"].mean(),
            "all": df["avg_vote"].mean()
        }

        st.session_state["baseline_director_metrics"] = baseline_metrics

init_baseline_director_rating_metrics(data)

def initialize_baseline_metrics(data):
    st.session_state["baseline_metrics"] = {
        "all_male": display_gender_composition_pct(data, gender="m"),
        "all_female": display_gender_composition_pct(data, gender="f"),
        "female_director": (data["director_1_gender"].str.lower() == "f").mean() * 100,
        "female_writer": (data["writer_1_gender"].str.lower() == "f").mean() * 100,
        "female_actor": (data["actor_1_gender"].str.lower() == "f").mean() * 100,
        "bechdel_3": (data["BD_rating"] == 3).mean() * 100
    }

# Helper for all-female/male composition
def display_gender_composition_pct(data, gender="f"):
    gender_cols = [
        "director_1_gender", "director_2_gender",
        "writer_1_gender", "writer_2_gender",
        "actor_1_gender", "actor_2_gender"
    ]
    filtered = data[gender_cols]
    total_valid = filtered.apply(lambda row: len([g for g in row if pd.notna(g)]) >= 3, axis=1).sum()

    def valid_and_consistent(row):
        known_genders = [g for g in row if pd.notna(g)]
        return len(known_genders) >= 3 and all(g == gender for g in known_genders)

    return (filtered.apply(valid_and_consistent, axis=1).sum() / total_valid) * 100 if total_valid > 0 else 0

initialize_baseline_metrics(data)




# Use the returned value to drive your view logic

if page == "Analysis" :
    plot_selected_distribution(filtered_data)

    plot_financial_distribution(filtered_data)

    plot_categorical_donut(filtered_data)

    
elif page == "Bechdel":

    st.markdown("#### Genre and Change over Time")
    mode_dist = st.selectbox(
        "Show Plots by...:",
        options=["Year", "Gendered Role"]
    )
    
    col3, col4 = st.columns(2)
    with col3 :

        if mode_dist == "Year" :
            bechdel_rating_per_year(filtered_data)
        else :
            bechdel_rating_by_gender_all_roles(filtered_data)

    with col4:
        if mode_dist == "Year":
            avg_bechdel_score_by_genre_and_year(filtered_data)
        else :
            avg_bechdel_score_by_genre_and_gendered_role(filtered_data)


    st.markdown("#### Budget Analysis")
    col5, col6 = st.columns(2)
    with col5 :
        income_and_budget_by_bechdel_pass(filtered_data)
        #income_by_bechdel_pass(filtered_data)
    with col6 :
        #budget_by_bechdel_pass(filtered_data)
        scatter_budget_vs_income(filtered_data)


    st.markdown("#### Rating Analysis")
    plot_bechdel_vs_rating(filtered_data)


elif page == "Industry":

    st.markdown("#### Female Representation")

    display_gender_composition_metrics(filtered_data)
    col5, col6 = st.columns(2)
    with col5 :
        #female_roles_overall(filtered_data)
        female_roles_over_time(filtered_data)
        
        
    with col6:
        female_roles_by_genre_heatmap(filtered_data)


    st.markdown("#### Finances and Rating")    
    compare_financials_by_role_gender(filtered_data)



elif page == "Tree" :
    st.markdown("#### Sunburst chart visualization of a decision tree")

    # Slider for selecting tree depth
    with top_sidebar :
        st.markdown("#### Page-Specific Filters")
        max_depth = st.slider("Max Tree Depth", min_value=2, max_value=20, value=3, step=1)
        min_leaf_size = st.slider("Minimum Leaf Size of Tree", min_value=1, max_value=300, value=60, step=1)

        selected_features = st.multiselect(
            "Select features to include in the model:",
            options=all_features,
            default=all_features
        )

    sunburst(filtered_data, max_depth=max_depth, selected_features=selected_features, min_samples_leaf=min_leaf_size)
    
elif page == "Wordcloud":
    st.header("Wordcloud", help="Use the filters to change the wordcloud, especially the female and male director, writer and actor roles.")
    with top_sidebar :
        st.markdown("#### Page-Specific Filters")
        max_words = st.slider("Max Words", min_value=100, max_value=1000, value=200, step=10)

    wordcloud(filtered_data, max_words=max_words)

    if st.checkbox("Animate me!") :
        bar_race_by_year(filtered_data)
    else :
        top_words_bar_chart(filtered_data, top_n=int(max_words/10))


elif page == "Dataset":
    st.header("Dataset")
    st.dataframe(filtered_data, use_container_width=True)


elif page == "Info":
    st.header("Info")

else:
    st.markdown("""
            #### Oh look an easteregg 🥚
            This is the original comic that introduced the Bechdel-Wallace test:
                """)
    st.image("https://i.kym-cdn.com/photos/images/newsfeed/000/718/730/2d6.jpg")
    st.markdown("oh and also")
    st.image("https://stackward.com/wp-content/uploads/2016/09/177.jpg")

    
#with st.expander("session state"):
    #st.json(st.session_state)





