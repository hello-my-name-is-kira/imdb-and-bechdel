import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go

primary_color = "#08DBE6"

def plot_selected_distribution(data: pd.DataFrame):
    selected_columns = {
        "year": "Year",
        "duration": "Duration",
        "avg_vote": "Average Vote",
        "votes": "Vote Count"
    }

    selected_col = st.selectbox("Select variable to visualize:", list(selected_columns.keys()), format_func=lambda x: selected_columns[x])

    df = data.copy()

    col1, col2 = st.columns([2, 1])

    if selected_col == "year":
        with col1:
            group_counts = df["year"].value_counts().sort_index()
            fig_bar = px.bar(
                x=group_counts.index,
                y=group_counts.values,
                labels={"x": "Year", "y": "Number of Movies"},
                title="Number of Movies per Year",
                color_discrete_sequence=[primary_color]
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_box = px.box(df, y="year", points="all", title="Boxplot of Release Years")
            fig_box.update_traces(marker_color=primary_color, line_color=primary_color)
            st.plotly_chart(fig_box, use_container_width=True)


    elif selected_col == "duration":
        with col1:
            bins = np.arange(0, df["duration"].max() + 10, 10)
            labels = [f"{int(b)}-{int(b+10)}" for b in bins[:-1]]
            df["duration_bin"] = pd.cut(df["duration"], bins=bins, labels=labels, right=False)
            group_counts = df["duration_bin"].value_counts().sort_index()
            fig_bar = px.bar(
                x=group_counts.index.astype(str),
                y=group_counts.values,
                labels={"x": "minutes", "y": "Number of Movies"},
                title="Distribution by Duration",
                color_discrete_sequence=[primary_color]
            )
            st.plotly_chart(fig_bar)

        with col2:
            fig_box = px.box(df, y="duration", points="all", title="Boxplot of Duration")
            fig_box.update_traces(marker_color=primary_color, line_color=primary_color)
            st.plotly_chart(fig_box)

    elif selected_col == "avg_vote":
        with col1:
            df["vote_bin"] = df["avg_vote"].apply(lambda x: int(x))
            group_counts = df["vote_bin"].value_counts().sort_index()
            fig_bar = px.bar(
                x=group_counts.index.astype(str),
                y=group_counts.values,
                labels={"x": "Capped IMDB Vote", "y": "Number of Movies"},
                title="Distribution of Average IMDB Votes",
                color_discrete_sequence=[primary_color]
            )
            st.plotly_chart(fig_bar)

        with col2:
            fig_box = px.box(
                df, y="avg_vote", 
                points="all", 
                title="Boxplot of Average Vote")
            
            fig_box.update_traces(marker_color=primary_color, line_color=primary_color)
            st.plotly_chart(fig_box)


    elif selected_col == "votes":
        with col1:
            df_votes = df[df["votes"] > 0].copy()
            
            bins = [0, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2000000]
            labels = [f"{int(bins[i]/1000)}K–{int(bins[i+1]/1000)}K" for i in range(len(bins)-1)]
            
            df_votes["vote_bin"] = pd.cut(df_votes["votes"], bins=bins, labels=labels, right=False)
            group_counts = df_votes["vote_bin"].value_counts().sort_index()
            
            group_counts = group_counts[group_counts > 0]
            
            fig_bar = px.bar(
                x=group_counts.index.astype(str),
                y=group_counts.values,
                labels={"x": "Vote Count Range", "y": "Number of Movies"},
                title="Distribution of Vote Counts",
                color_discrete_sequence=[primary_color]
            )
            st.plotly_chart(fig_bar)

        with col2:
            fig_box = px.box(df_votes, y="votes", points="all", log_y=True, title="Vote Count on Log Scale")
            fig_box.update_traces(marker_color=primary_color, line_color=primary_color)

            st.plotly_chart(fig_box)

def plot_financial_distribution(data: pd.DataFrame):
    df = data.copy()

    #st.markdown(df.columns)
    financial_columns = {
        "budget": "Budget",
        "usa_gross_income": "USA Gross Income",
        "worlwide_gross_income": "Worldwide Gross Income"
    }

    selected_col = st.selectbox(
        "Select financial metric to visualize:",
        options=list(financial_columns.keys()),
        format_func=lambda x: financial_columns[x]
    )

    df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')
    df = df[df[selected_col] > 0]

    col1, col2 = st.columns([2, 1])

    with col1:
        yearly_stats = df.groupby("year")[selected_col].agg(["mean", "median"]).reset_index()

        fig_line = go.Figure()

        fig_line.add_trace(go.Scatter(
            x=yearly_stats["year"],
            y=yearly_stats["mean"],
            mode='lines+markers',
            name='Mean',
            line=dict(color="#3AFF7F")
        ))

        fig_line.add_trace(go.Scatter(
            x=yearly_stats["year"],
            y=yearly_stats["median"],
            mode='lines+markers',
            name='Median',
            line=dict(color=primary_color, dash='dash')
        ))

        fig_line.update_layout(
            title=f"{financial_columns[selected_col]} Over Time (Log Scale)",
            xaxis_title="Year",
            yaxis_title=financial_columns[selected_col],
            yaxis_type="log"
        )

        st.plotly_chart(fig_line)

    with col2:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=df[selected_col],
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            name=financial_columns[selected_col]
        ))

        fig_box.update_layout(
            title=f"Boxplot of {financial_columns[selected_col]}",
            yaxis_type="log"
        )
        fig_box.update_traces(marker_color=primary_color, line_color=primary_color)

        st.plotly_chart(fig_box)

def plot_categorical_donut(data: pd.DataFrame):
    df = data.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    categorical_columns = {
        "genre_1": "Primary Genre",
        "genre_2": "Secondary Genre",
        "genre_3": "Tertiary Genre",
        "language_1": "Language",
        "country": "Country",
        "bd_rating": "Bechdel Test Score",
        "director_1_gender": "Director 1 Gender",
        "director_2_gender": "Director 2 Gender",
        "writer_1_gender": "Writer 1 Gender",
        "writer_2_gender": "Writer 2 Gender",
        "actor_1_gender": "Actor 1 Gender",
        "actor_2_gender": "Actor 2 Gender"
    }

    sel_col, chk_col = st.columns([2, 1])
    with sel_col:
        selected_col = st.selectbox(
            "Select a categorical variable:",
            options=list(categorical_columns.keys()),
            format_func=lambda x: categorical_columns[x]
        )
    with chk_col:
        show_missing = st.checkbox("Show missing/unknown", help="groups that contain less than 5% of the data are grouped into 'other'")

    series = df[selected_col]
    if "gender" in selected_col:
        valid_values = ["m", "f"]
        series = series.where(series.isin(valid_values))

    # Add missing/unknown if selected
    if show_missing:
        series = series.fillna("missing / unknown")
    else:
        series = series.dropna()

    # Compute counts and percentages
    value_counts = series.value_counts()
    total = value_counts.sum()
    percentages = (value_counts / total) * 100

    # Only group <5% if there are more than 6 distinct categories
    if len(value_counts) > 6:
        grouped = value_counts.copy()
        grouped[percentages < 5] = 0
        grouped = grouped[grouped > 0]
        other_sum = value_counts[percentages < 5].sum()
        if other_sum > 0:
            grouped["other"] = other_sum
    else:
        grouped = value_counts

    # Final dataframe for plot and table
    plot_df = grouped.reset_index()
    plot_df.columns = [selected_col, "count"]

    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        fig = px.pie(
            plot_df,
            names=selected_col,
            values="count",
            title=f"Distribution of {categorical_columns[selected_col]}",
            hole=0.4
        )

        fig.update_traces(textinfo='label')

        if "missing / unknown" in plot_df[selected_col].values:
            fig.update_traces(
                marker=dict(colors=[
                    "lightgray" if val == "missing / unknown" else None
                    for val in plot_df[selected_col]
                ])
            )

        st.plotly_chart(fig)


    with table_col:
        plot_df = grouped.reset_index()
        plot_df.columns = [categorical_columns[selected_col], "Count"] 

        st.write("Value Counts:")
        st.dataframe(plot_df)

