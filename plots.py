import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression


pass_color = "#099d49"
fail_color = "#DB5D5D"
score2_color = "#ffd86c"
woman_color = "#6b0869"
man_color = "#08306b"
mix_color = "#17086b"

def gender_map(df) :
    gender_map = {
        "m": "male",
        "f": "female",
        "u": "unknown"
    }
    gender_cols = [
        "director_1_gender",
        "writer_1_gender", "writer_2_gender",
        "actor_1_gender", "actor_2_gender"
    ]

    for col in gender_cols:
        if col in df.columns:
            df[col] = df[col].map(gender_map)

    return df

def income_and_budget_by_bechdel_pass(data):
    df = data.copy()

    df_pass = df[df["BD_rating"] == 3]
    df_fail = df[df["BD_rating"] < 3]

    for col in ["usa_gross_income", "worlwide_gross_income", "budget"]:
        df_pass[col] = pd.to_numeric(df_pass[col], errors="coerce")
        df_fail[col] = pd.to_numeric(df_fail[col], errors="coerce")

    avg_values = pd.DataFrame({
        "Group": ["Pass", "Fail"],
        "USA Gross": [df_pass["usa_gross_income"].mean(), df_fail["usa_gross_income"].mean()],
        "Worldwide Gross": [df_pass["worlwide_gross_income"].mean(), df_fail["worlwide_gross_income"].mean()],
        "Budget": [df_pass["budget"].mean(), df_fail["budget"].mean()]
    })

    avg_melt = avg_values.melt(id_vars="Group", var_name="Category", value_name="Average ($)")

    avg_melt["hover"] = avg_melt.apply(
        lambda row: (
            f"For <b>{row['Category']}</b>, movies that "
            f"<b>{'passed' if row['Group']=='Pass' else 'failed'}</b> the Bechdel Test "
            f"had an average of <b>${row['Average ($)']:,.0f}</b>"
        ),
        axis=1
    )

    color_map = {
        "Pass": pass_color,
        "Fail": fail_color
    }

    fig = px.bar(
        avg_melt,
        x="Category",
        y="Average ($)",
        color="Group",
        barmode="group",
        title="Income and Budget by Bechdel Pass/Fail",
        labels={"Category": "Metric", "Group": "Bechdel Test"},
        color_discrete_map=color_map
    )

    for trace in fig.data:
        group_name = trace.name 
        group_df = avg_melt[avg_melt["Group"] == group_name]
        hover_texts = [
            group_df[group_df["Category"] == cat]["hover"].values[0]
            for cat in group_df["Category"]
        ]
        trace.customdata = hover_texts
        trace.hovertemplate = "%{customdata}<extra></extra>"

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True, key="bd_income_budget")


def bechdel_rating_per_year(data) :

    bd = data[data["BD_rating"].notna()].copy()
    total_per_year = bd.groupby("year").size().reset_index(name="total")

    bd_counts = (
        bd.groupby(["year", "BD_rating"])
        .size()
        .reset_index(name="count")
        .merge(total_per_year, on="year")
    )
    bd_counts["Percentage"] = (bd_counts["count"] / bd_counts["total"]) * 100
    bd_counts["BD_rating"] = bd_counts["BD_rating"].astype(int)

    pivot = bd_counts.pivot(index="year", columns="BD_rating", values="Percentage").fillna(0)

    for rating in [0, 1, 2, 3]:
        if rating not in pivot.columns:
            pivot[rating] = 0
    pivot = pivot[[3, 2, 1, 0]]

    colors = {
        3: mix_color,
        2: "#866bd6",
        1: "#d6c6ef",
        0: "#ffffff"
    }
    labels = {
        3: "passed (3)",
        2: "2",
        1: "1",
        0: "0"
    }

    fig = go.Figure()


    for rating in [3, 2, 1, 0]:
        hover_texts = [
            f"In the year <b>{year}</b>, <b>{val:.1f}%</b> of movies had a Bechdel rating of <b>{rating}</b>"
            for year, val in zip(pivot.index, pivot[rating])
        ]

        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[rating],
            name=labels[rating],
            marker_color=colors[rating],
            legendgroup=str(rating),
            showlegend=True,
            hovertext=hover_texts,
            hoverinfo="text"
        ))

    fig.update_layout(
        barmode='stack',
        title="Bechdel Rating Distribution per Year",
        xaxis_title="Year",
        yaxis_title="Percentage",
        template="plotly_white",
        legend=dict(
            traceorder="reversed" 
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def bechdel_rating_by_gender_all_roles(data):
    role_map = {
        "Writer": "writer_1_gender",
        "Director": "director_1_gender",
        "Actor": "actor_1_gender"
    }

    df = data.copy()
    df = df[df["BD_rating"].notna()]
    df["BD_rating"] = df["BD_rating"].astype(int)

    records = []
    for role_label, col in role_map.items():
        if col not in df.columns:
            continue
        role_df = df[df[col].notna()]
        role_df[col] = role_df[col].astype(str).str.lower().str.strip()
        role_df = role_df[role_df[col].isin(["f", "m"])]

        total_per_gender = role_df.groupby(col).size().reset_index(name="total")

        counts = (
            role_df.groupby([col, "BD_rating"])
            .size()
            .reset_index(name="count")
            .merge(total_per_gender, on=col)
        )
        counts["Percentage"] = (counts["count"] / counts["total"]) * 100
        counts["Role"] = role_label
        counts["Gender"] = counts[col].map({"f": "Female", "m": "Male"})
        records.append(counts[["Role", "Gender", "BD_rating", "Percentage"]])

    full_df = pd.concat(records, ignore_index=True)

    pivot = full_df.pivot_table(
        index=["Role", "Gender"],
        columns="BD_rating",
        values="Percentage",
        fill_value=0
    ).reset_index()


    for rating in [0, 1, 2, 3]:
        if rating not in pivot.columns:
            pivot[rating] = 0

    pivot = pivot[["Role", "Gender", 3, 2, 1, 0]]  


    fig = go.Figure()

    for rating in [3, 2, 1, 0]:
        for gender in ["Female", "Male"]:
            subset = pivot[pivot["Gender"] == gender]
            role_labels = subset["Role"] + " - " + gender

            hover_texts = [
                f"For <b>{gender}</b> in role <b>{role}</b>, <b>{val:.1f}%</b> of movies had a Bechdel rating of <b>{rating}</b>"
                for role, val in zip(subset["Role"], subset[rating])
            ]

            fig.add_trace(go.Bar(
                x=role_labels,
                y=subset[rating],
                name=f"{rating} ({gender})" if rating != 3 else f"passed (3) ({gender})",
                marker_color=(
                    {
                        "Female": {
                            3: woman_color,
                            2: "#d66bcd",
                            1: "#efc6ee",
                            0: "#ffffff"
                        },
                        "Male": {
                            3: man_color,
                            2: "#6baed6",
                            1: "#c6dbef",
                            0: "#ffffff"
                        }
                    }[gender][rating]
                ),
                legendgroup=f"{rating}-{gender}",
                showlegend=(gender == "Female"),
                hovertext=hover_texts,
                hoverinfo="text"
            ))


    fig.update_layout(
        barmode="stack",
        title="Bechdel Rating Distribution by Gender and Role",
        xaxis_title="Role and Gender",
        yaxis_title="Percentage",
        template="plotly_white",
        legend=dict(traceorder="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)


def avg_bechdel_score_by_genre_and_year(data) :
    df = gender_map(data.copy())    

    genre_year_df = (
        df.explode("all_genres")[["year", "all_genres", "BD_rating"]]
        .dropna()
        .query("2000 <= year < 2020")
    )

    grouped = (
        genre_year_df
        .groupby(["all_genres", "year"])
        .agg(
            avg_score=("BD_rating", "mean"),
            count=("BD_rating", "size")
        )
        .reset_index()
    )

    top_genres = (
        df.explode("all_genres")["all_genres"]
        .value_counts()
        .head(10)
        .index
    )
    grouped = grouped[grouped["all_genres"].isin(top_genres)]

    grouped["hover"] = grouped.apply(
        lambda row: (
            f"In the year <b>{row['year']}</b>, "
            f"<b>{row['count']}</b> movies contributed to a Bechdel rating of "
            f"<b>{row['avg_score']:.2f}</b> for the genre <b>{row['all_genres']}</b>"
        ),
        axis=1
    )

    pivot_avg = grouped.pivot(index="all_genres", columns="year", values="avg_score")
    pivot_hover = grouped.pivot(index="all_genres", columns="year", values="hover")

    custom_colors = [
        [0.0, "#000000"],  
        [0.33, "#d73027"], 
        [0.66, score2_color], 
        [1.0, pass_color] 
    ]

    fig = px.imshow(
        pivot_avg,
        x=pivot_avg.columns,
        y=pivot_avg.index,
        color_continuous_scale=custom_colors,
        zmin=0,
        zmax=3,
        text_auto=False,
        labels={"color": "Avg Score"},
        title="Average Bechdel Score by Genre and Year"
    )

    fig.update_traces(customdata=pivot_hover.values, hovertemplate="%{customdata}<extra></extra>")

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Genre",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True, key="bd_heatmap")


def avg_bechdel_score_by_genre_and_gendered_role(data):
    role_map = {
        "Writer": "writer_1_gender",
        "Director": "director_1_gender",
        "Actor": "actor_1_gender"
    }

    df = data.copy()
    df = df[df["BD_rating"].notna()]
    df["BD_rating"] = df["BD_rating"].astype(int)
    df = df.explode("all_genres").dropna(subset=["all_genres"])

    records = []

    for role_label, col in role_map.items():
        if col not in df.columns:
            continue

        role_df = df.dropna(subset=[col])
        role_df[col] = role_df[col].astype(str).str.lower().str.strip()
        role_df = role_df[role_df[col].isin(["f", "m"])]

        role_df["Gender"] = role_df[col].map({"f": "Female", "m": "Male"})
        role_df["Role"] = role_label
        role_df["Role-Gender"] = role_df["Role"] + " - " + role_df["Gender"]

        grouped = (
            role_df
            .groupby(["all_genres", "Role-Gender"])["BD_rating"]
            .agg(avg_score="mean", count="size")
            .reset_index()
        )

        records.append(grouped)

    full_df = pd.concat(records, ignore_index=True)

    top_genres = (
        df["all_genres"]
        .value_counts()
        .head(10)
        .index
    )
    full_df = full_df[full_df["all_genres"].isin(top_genres)]

    full_df["hover"] = full_df.apply(
        lambda row: (
            f"For the genre <b>{row['all_genres']}</b>, "
            f"<b>{row['count']}</b> movies contributed to a Bechdel rating of "
            f"<b>{row['avg_score']:.2f}</b> for the role <b>{row['Role-Gender']}</b>"
        ),
        axis=1
    )

    pivot_avg = full_df.pivot(index="all_genres", columns="Role-Gender", values="avg_score").fillna(0)
    pivot_hover = full_df.pivot(index="all_genres", columns="Role-Gender", values="hover").fillna("")

    desired_order = [
        "Actor - Female", "Director - Female", "Writer - Female",
        "Actor - Male", "Director - Male", "Writer - Male"
    ]

    existing_columns = [col for col in desired_order if col in pivot_avg.columns]
    pivot_avg = pivot_avg[existing_columns]
    pivot_hover = pivot_hover[existing_columns]

    custom_colors = [
        [0.0, "#000000"],  
        [0.33, "#d73027"],  
        [0.66, score2_color],
        [1.0, pass_color] 
    ]

    fig = px.imshow(
        pivot_avg,
        x=pivot_avg.columns,
        y=pivot_avg.index,
        color_continuous_scale=custom_colors,
        zmin=0,
        zmax=3,
        text_auto=False,
        labels={"color": "Avg Score"},
        title="Average Bechdel Score by Genre and Gendered Role"
    )

    fig.update_traces(
        customdata=pivot_hover.values,
        hovertemplate="%{customdata}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Gendered Role",
        yaxis_title="Genre",
        template="plotly_white",
        xaxis=dict(tickangle=45)
    )

    st.plotly_chart(fig, use_container_width=True, key="bd_genre_role_heatmap")


def female_roles_over_time(data) :
    df = gender_map(data.copy())    

    role_cols = {
        'Director 1': 'director_1_gender',
        'Writer 1': 'writer_1_gender',
        'Writer 2': 'writer_2_gender',
        'Actor 1': 'actor_1_gender',
        'Actor 2': 'actor_2_gender'
    }

    df = data[(data['year'] >= 2000) & (data['year'] <= 2020)].copy()

    for col in role_cols.values():
        df[col] = df[col].astype(str).str.strip().str.lower()

    records = []
    for role_label, col in role_cols.items():
        temp = df[['year', col]].dropna()
        temp.columns = ['year', 'gender']
        temp['role'] = role_label
        records.append(temp)

    long_df = pd.concat(records, ignore_index=True)

    grouped_raw = (
        long_df.groupby(['year', 'role', 'gender'])
        .size()
        .reset_index(name='count')
    )

    grouped = grouped_raw.pivot_table(
        index=['year', 'role'],
        columns='gender',
        values='count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    grouped['f'] = grouped.get('f', 0)
    grouped['m'] = grouped.get('m', 0)
    grouped['Total'] = grouped['f'] + grouped['m']
    grouped['% Female'] = (grouped['f'] / grouped['Total']) * 100

    fig = px.line(
        grouped,
        x='year',
        y='% Female',
        color='role',
        markers=True,
        title="Female Representation in Key Roles"
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Female Percentage (%)",
        template="plotly_white",
        yaxis=dict(range=[0, 100]),
        legend_title="Role"
    )

    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        annotation_text="Gender Parity (50%)",
        annotation_position="top left"
    )

    st.plotly_chart(fig, use_container_width=True, key="female_roles_chart")


def female_roles_by_genre_heatmap(data):
    role_cols = {
        'Director 1': 'director_1_gender',
        'Writer 1': 'writer_1_gender',
        'Writer 2': 'writer_2_gender',
        'Actor 1': 'actor_1_gender',
        'Actor 2': 'actor_2_gender'
    }

    df = data[(data["year"] >= 2000) & (data["year"] <= 2020)].copy()
    df = df[df["genre_1"].notna()]

    genre_counts = df["genre_1"].value_counts()
    valid_genres = genre_counts[genre_counts >= 4].index
    df = df[df["genre_1"].isin(valid_genres)]

    records = []
    for role_label, col in role_cols.items():
        subset = df[["genre_1", col]].dropna()
        subset[col] = subset[col].astype(str).str.strip().str.lower()

        grouped = (
            subset.groupby("genre_1")[col]
            .apply(lambda x: (x == "f").sum() / ((x == "f").sum() + (x == "m").sum()) * 100 if ((x == "f") | (x == "m")).any() else None)
            .reset_index(name="Female %")
        )
        grouped["Role"] = role_label
        records.append(grouped)

    heatmap_df = pd.concat(records, ignore_index=True)

    count_records = []
    for role_label, col in role_cols.items():
        subset = df[["genre_1", col]].dropna()
        subset[col] = subset[col].astype(str).str.strip().str.lower()
        grouped = (
            subset.groupby("genre_1")[col]
            .apply(lambda x: ((x == "f") | (x == "m")).sum())
            .reset_index(name="Count")
        )
        grouped["Role"] = role_label
        count_records.append(grouped)

    count_df = pd.concat(count_records, ignore_index=True)

    z_matrix = heatmap_df.pivot(index="Role", columns="genre_1", values="Female %")
    count_matrix = count_df.pivot(index="Role", columns="genre_1", values="Count")

    role_order = ["Actor 1", "Actor 2", "Director 1", "Writer 1", "Writer 2"]
    heatmap_df["Role"] = pd.Categorical(heatmap_df["Role"], categories=role_order, ordered=True)
    heatmap_df = heatmap_df.sort_values(["genre_1", "Role"])

    hover_texts = []
    for role in z_matrix.index:
        row_texts = []
        for genre in z_matrix.columns:
            pct = z_matrix.loc[role, genre]
            count = count_matrix.loc[role, genre]
            if pd.isna(pct):
                row_texts.append("No data for this combination.")
            else:
                row_texts.append(
                    f"In the genre <b>{genre}</b>, <b>{pct:.1f}%</b> of <b>{int(count)}</b> movies "
                    f"had a female <b>{role}</b>."
                )
        hover_texts.append(row_texts)
    hover_texts = np.array(hover_texts)

    custom_colorscale = [
        [0.0, fail_color],
        [0.5, "#f5f5dc"],  
        [1.0, pass_color]
    ]

    fig = px.imshow(
        z_matrix,
        color_continuous_scale=custom_colorscale,
        zmin=0, zmax=100,
        text_auto=".1f",
        labels={"color": "Female %"},
        title="Female Representation in Key Roles by Genre"
    )

    fig.update_traces(
        customdata=hover_texts,
        hovertemplate="%{customdata}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Primary Genre",
        yaxis_title="Role",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


def scatter_budget_vs_income(data):
    if "budget" not in data.columns or "worlwide_gross_income" not in data.columns or "BD_rating" not in data.columns:
        st.warning("Missing columns: 'budget', 'worlwide_gross_income', or 'BD_rating'")
        return

    df = data.copy()
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df["income"] = pd.to_numeric(df["worlwide_gross_income"], errors="coerce")
    df["BD_rating"] = pd.to_numeric(df["BD_rating"], errors="coerce")

    df = df.dropna(subset=["budget", "income", "BD_rating"])
    df = df[(df["budget"] > 0) & (df["income"] > 0)]

    df["bechdel_result"] = df["BD_rating"].apply(lambda x: "Pass" if x >= 3 else "Fail")

    fig = px.scatter(
        df,
        x="budget",
        y="income",
        color="bechdel_result",
        color_discrete_map={
            "Pass": pass_color,  
            "Fail": fail_color   
        },
        hover_data=["title", "year"] if "title" in df.columns and "year" in df.columns else None,
        title="Worldwide Income vs Budget",
        labels={
            "budget": "Budget ($)",
            "income": "Worldwide Gross Income ($)",
            "bechdel_result": "Bechdel Test"
        },
        log_x=True,
        log_y=True
    )

    fig.update_traces(marker=dict(size=6, opacity=0.7))

    fig.add_trace(go.Scatter(
        x=df["budget"],
        y=df["budget"],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Break-even Line",
        showlegend=True
    ))

    def add_regression_line(fig, df_subset, label, color):
        x_log = np.log10(df_subset["budget"].values).reshape(-1, 1)
        y_log = np.log10(df_subset["income"].values)

        model = LinearRegression().fit(x_log, y_log)
        x_vals = np.linspace(x_log.min(), x_log.max(), 100).reshape(-1, 1)
        y_preds = model.predict(x_vals)

        x_vals_real = 10 ** x_vals.flatten()
        y_vals_real = 10 ** y_preds

        fig.add_trace(go.Scatter(
            x=x_vals_real,
            y=y_vals_real,
            mode="lines",
            name=f"{label} Regression",
            line=dict(color=color, dash="dot")
        ))

    df_pass = df[df["bechdel_result"] == "Pass"]
    df_fail = df[df["bechdel_result"] == "Fail"]

    if not df_pass.empty:
        add_regression_line(fig, df_pass, "Pass", pass_color)
    if not df_fail.empty:
        add_regression_line(fig, df_fail, "Fail", fail_color)

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def plot_bechdel_vs_rating(data):
    df = data.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "avg_vote" not in df.columns or "bd_rating" not in df.columns:
        st.warning("Missing columns: 'avg_vote' or 'BD_rating'")
        return

    df["avg_vote"] = pd.to_numeric(df["avg_vote"], errors="coerce")
    df["bd_rating"] = pd.to_numeric(df["bd_rating"], errors="coerce")
    df = df.dropna(subset=["avg_vote", "bd_rating"])

    grouped = df.groupby("bd_rating").agg(
        avg_rating=("avg_vote", "mean"),
        count=("avg_vote", "count")
    ).reset_index()

    col1, col2 = st.columns([1, 2])

    color_map = {
        0: fail_color,
        1: fail_color,
        2: score2_color,
        3: pass_color
    }

    df["bd_rating_str"] = df["bd_rating"].astype(int).astype(str)

    with col1:
        fig_box = px.box(
            df,
            x="bd_rating_str",
            y="avg_vote",
            points="all",
            color="bd_rating_str",  
            color_discrete_map={
                "0": fail_color,
                "1": fail_color,
                "2": score2_color,
                "3": pass_color
            },
            title="Rating Distribution by Bechdel Score",
            labels={"bd_rating_str": "Bechdel Score", "avg_vote": "Rating"},
            range_y=[0, 10]
        )

        st.plotly_chart(fig_box)

    with col2:
            grouped["hover"] = grouped.apply(
                lambda row: (
                    f"<b>{int(row['count'])}</b> movies contributed to an average IMDb rating of "
                    f"<b>{row['avg_rating']:.1f}</b> for the Bechdel score of <b>{int(row['bd_rating'])}</b>"
                ),
                axis=1
            )


            fig_bar = px.bar(
                grouped,
                x="bd_rating",
                y="avg_rating",
                color="count",
                color_continuous_scale="Teal",
                labels={
                    "bd_rating": "Bechdel Score",
                    "avg_rating": "Average Rating",
                    "count": "Movie Count"
                },
                title="Average Rating per Bechdel Score",
                range_y=[0, 10]
            )

            fig_bar.update_traces(
                customdata=grouped["hover"],
                hovertemplate="%{customdata}<extra></extra>"
            )

            fig_bar.update_layout(
                xaxis=dict(
                    dtick=1,
                    tickmode='linear'
                ),
                template="plotly_white"
            )

            st.plotly_chart(fig_bar, use_container_width=True)


def compare_financials_by_role_gender(data):
    color_map = {
        "Female": woman_color,
        "Male": man_color
    }

    
    role_columns = {
        "Director": "director_1_gender",
        "Writer": "writer_1_gender",
        "Actor": "actor_1_gender"
    }

    view_mode = st.radio(
        "Select view:",
        options=["Simple View", "Advanced View"],
        horizontal=True
    )

    financial_metrics = {
        "Budget": "budget",
        "Worldwide Gross Income": "worlwide_gross_income",
        "Profit": None  
    }

    col1, col2, col3 = st.columns(3)

    for i, (label, col) in enumerate(financial_metrics.items()):
        df = data.copy()

        if label == "Profit":
            df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
            df["worlwide_gross_income"] = pd.to_numeric(df["worlwide_gross_income"], errors="coerce")
            df["profit"] = df["worlwide_gross_income"] - df["budget"]
            value_col = "profit"
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            value_col = col

        df = df.dropna(subset=[value_col])

        records = []
        for role, gender_col in role_columns.items():
            if gender_col not in df.columns:
                continue

            role_df = df.dropna(subset=[gender_col])
            role_df[gender_col] = role_df[gender_col].astype(str).str.lower().str.strip()

            for gender_code in ["f", "m"]:
                filtered = role_df[role_df[gender_col] == gender_code].copy()
                filtered["Role"] = role
                filtered["Gender"] = "Female" if gender_code == "f" else "Male"
                filtered["Value"] = filtered[value_col]
                records.append(filtered[["Value", "Role", "Gender"]])

        plot_df = pd.concat(records, ignore_index=True)

        if view_mode == "Advanced View":
            fig = px.box(
                plot_df,
                x="Role",
                y="Value",
                color="Gender",
                points="outliers",
                title=f"{label} Distribution by Role and Gender",
                labels={"Value": f"{label} ($)"},
                color_discrete_map=color_map
            )
            fig.update_yaxes(type="log")
        else:  
            agg_df = plot_df.groupby(["Role", "Gender"]).agg(Avg_Value=("Value", "mean")).reset_index()
            agg_df["hover"] = agg_df.apply(
                lambda row: (
                    f"For primary "
                    f"{'female' if row['Gender'].lower() == 'female' else 'male'} "
                    f"<b>{row['Role']}s</b>,<br>"
                    f"movies had an average {label.lower()} of <b>${row['Avg_Value']:,.0f}</b>"
                ),
                axis=1
            )

            fig = px.bar(
                agg_df,
                x="Role",
                y="Avg_Value",
                color="Gender",
                barmode="group",
                title=f"Average {label} by Role and Gender",
                labels={"Avg_Value": f"Average {label} ($)"},
                color_discrete_map=color_map
            )

            for trace in fig.data:
                gender = trace.name 
                subset = agg_df[agg_df["Gender"] == gender]
                hover_texts = [
                    subset[subset["Role"] == role]["hover"].values[0]
                    for role in subset["Role"]
                ]
                trace.customdata = hover_texts
                trace.hovertemplate = "%{customdata}<extra></extra>"

        if i == 0:
            with col1: st.plotly_chart(fig, use_container_width=True)
        elif i == 1:
            with col2: st.plotly_chart(fig, use_container_width=True)
        else:
            with col3: st.plotly_chart(fig, use_container_width=True)

    df = data.dropna(subset=["avg_vote"]).copy()

    records = []
    for role, gender_col in role_columns.items():
        if gender_col not in df.columns:
            continue

        role_df = df.dropna(subset=[gender_col])
        role_df[gender_col] = role_df[gender_col].astype(str).str.lower().str.strip()

        for gender_code in ["f", "m"]:
            filtered = role_df[role_df[gender_col] == gender_code].copy()
            filtered["Role"] = role
            filtered["Gender"] = "Female" if gender_code == "f" else "Male"
            filtered["IMDb Rating"] = filtered["avg_vote"]
            records.append(filtered[["IMDb Rating", "Role", "Gender"]])

    plot_df = pd.concat(records, ignore_index=True)

    if view_mode == "Advanced View":
        fig = px.box(
            plot_df,
            x="Role",
            y="IMDb Rating",
            color="Gender",
            points="outliers",
            title="IMDb Rating Distribution by Role and Gender",
            labels={"IMDb Rating": "IMDb Rating"},
            color_discrete_map=color_map
        )
    else: 
        agg_df = plot_df.groupby(["Role", "Gender"]).agg(Avg_Rating=("IMDb Rating", "mean")).reset_index()

        agg_df["hover"] = agg_df.apply(
            lambda row: (
                f"For primary {'female' if row['Gender'] == 'Female' else 'male'} "
                f"<b>{row['Role']}s</b>,<br>"
                f"movies had an average IMDb rating of <b>{row['Avg_Rating']:.1f}</b>"
            ),
            axis=1
        )

        fig = px.bar(
            agg_df,
            x="Role",
            y="Avg_Rating",
            color="Gender",
            barmode="group",
            title="Average IMDb Rating by Role and Gender",
            labels={"Avg_Rating": "Average IMDb Rating"},
            color_discrete_map=color_map
        )

        for trace in fig.data:
            gender = trace.name  
            subset = agg_df[agg_df["Gender"] == gender]
            hover_texts = [
                subset[subset["Role"] == role]["hover"].values[0]
                for role in subset["Role"]
            ]
            trace.customdata = hover_texts
            trace.hovertemplate = "%{customdata}<extra></extra>"

    col3, col4 = st.columns([2, 1])
    with col3:
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.empty()
