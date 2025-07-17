import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz
from sklearn.model_selection import train_test_split
from streamlit_echarts import st_echarts
import re
import plotly.graph_objects as go



def get_ratio_color(pass_count, fail_count, class_value):

    if pass_count == fail_count :
        return "#afafaf"
    elif class_value == 1:        
        return "#b3e5c7"
    elif class_value == 0 :
        return "#d7b7b5"
        

def clean_node_label(feature_name, threshold, direction: bool):
    #return feature_name
    base = feature_name.strip().lower()
    #value = f"{threshold:.2f}"

    # Handle binary gender cases
    if "is male" in base:
        person = "Actor 1" if "actor 1" in base else \
                 "Actor 2" if "actor 2" in base else \
                 "Writer 1" if "writer 1" in base else \
                 "Director" if "director" in base else "Person"
        gender = "Female" if direction is True else "Male"
        return f"{person} is {gender}"

    # Handle genre categories
    elif "genre" in base:
        parts = base.split()
        genre_type = parts[0].capitalize() + " Genre"
        genre_name = parts[2].capitalize()
        if direction is True:
            return f"{genre_type} is not {genre_name}"
        else:
            return f"{genre_type} is {genre_name}"
        
    elif "year" in base:
        parts = base.split()
        if len(parts) == 3:
            year_float = float(parts[2])
            if direction is True:
                return f"Released before {year_float}"
            else:
                return f"Released after {year_float}"
            
    # Duration
    elif "duration" in base:
        parts = base.split()
        if len(parts) == 3:
            duration = float(parts[2])
            return (
                f"Shorter than {int(duration)} min"
                if direction else
                f"Longer than {int(duration)} min"
            )

    # Avg Vote
    elif "avg vote" in base:
        parts = base.split()
        if len(parts) == 4:
            rating = float(parts[3])
            return (
                f"Rating is worse than {rating:.2f}" if direction else f"Rating is better than {rating:.2f}"
            )

    elif "votes" in base:
        parts = base.split()
        if len(parts) == 3:
            vote_amount = float(parts[2])
            return (
                f"Fewer Votes than {int(vote_amount)}"
                if direction else
                f"More Votes than {int(vote_amount)}"
            )
        
    elif "reviews from users" in base:
        parts = base.split()
        if len(parts) == 5:
            rating = float(4)
            return (
                f"Less than {int(rating)} Reviews"
                if direction else
                f"More than {int(rating)} Reviews"
            )

    else:
        if direction is True :
            op = "≤"
        else:
            op = ">"

        # Replace any existing operator with the correct one
        base = re.sub(r"(<=|≥|>=|<|>|=|≤)", op, feature_name)
        return base


all_features = [
    'year', 'duration', 'avg_vote', 'votes',
    'language_1', 'reviews_from_users',
    'worlwide_gross_income', 'usa_gross_income',
    'director_1_gender', 'writer_1_gender',
    'actor_1_gender', 'actor_2_gender',
    'genre_1', 'genre_2'
]

def train_bechdel_tree(data: pd.DataFrame, features: list = all_features, max_depth: int = 3, min_samples_leaf: int = 60):
    """
    Prepares the Bechdel dataset and trains a decision tree classifier.
    Returns the trained model, encoded feature matrix X, labels y, and tree structure.
    """
    df = data.copy()
    df['bechdel_pass'] = (df['BD_rating'] == 3).astype(int)

    df = df[features + ['bechdel_pass']].dropna()

    for col in ['worlwide_gross_income', 'usa_gross_income']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"${int(round(float(str(x).replace('$', '').replace(',', '').strip())))}"
                if pd.notna(x) else np.nan
            )

    df_encoded = pd.get_dummies(df.drop(columns=['bechdel_pass']), drop_first=True)
    X = df_encoded
    y = df['bechdel_pass']

    clean_names = {
        col: col.replace('_', ' ').replace('gender m', 'is male')
               .replace('genre 1', 'Primary Genre')
               .replace('genre 2', 'Secondary Genre')
               .title()
        for col in X.columns
    }
    X = X.rename(columns=clean_names)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, 
        #class_weight='balanced', 
        random_state=42)
    clf.fit(X_train, y_train)

    return clf, X, y


def format_ancestry_tooltip(path, final_label=None, init_samples_parent=0):
    """
    Build a multiline tooltip from a list of (label, samples, gini, pass, fail) tuples.
    Optionally appends a final classification label.
    """
    filtered = path[1:] if path and path[0][0] == "Node 0" else path

    lines = []
    samples_parent = init_samples_parent  # initialize with None or 0

    for item in filtered:
        if len(item) == 3:
            label, samples, gini = item
            pass_fail = ""
        else:
            label, samples, gini, pass_count, fail_count = item
            pass_fail = f", Pass: {pass_count}, Fail: {fail_count}"

        parent_text = f" of {samples_parent}" if samples_parent is not None else ""
        lines.append(f"{label}, ({samples} samples {parent_text}), gini: {gini:.3f}{pass_fail}")

        # update for next iteration
        samples_parent = samples


    if final_label:
        lines.append(final_label)

    return "<br/>".join(lines)


def build_sunburst_json(tree, node_id, X_cols, max_depth, current_depth=0,
                        parent_name=None, sample_amount=None, direction = None,
                        path=None, parent_gini=None, parent_values=None, parent_samples=None):

    """
    Builds the json for the suburst chart
    recursive function that calls itself twice for each iteration
    layers are shifted because basic tree data is unfit for sunburst chart
    """
    children_left = tree.children_left
    children_right = tree.children_right
    thresholds = tree.threshold
    features = tree.feature
    values = tree.value
    total_samples = tree.n_node_samples
    impurities = tree.impurity

    class_id = bool(np.argmax(values[node_id][0]))

    if path is None:
        path = []

    if sample_amount is None:
        sample_amount = int(total_samples[node_id])

    feature_name = X_cols[features[node_id]] if features[node_id] >= 0 else None
    threshold = thresholds[node_id] if features[node_id] >= 0 else None

    pass_count = 0
    fail_count = 0
    class_id = None

    if parent_values is not None and parent_samples is not None :
        class_id = int(np.argmax(values[node_id][0]))

        pass_count = int(round(values[node_id][0][0]*sample_amount))
        fail_count = int(round(values[node_id][0][1]*sample_amount))

        color = get_ratio_color(pass_count, fail_count, np.argmax(values[node_id][0]))
    else :
        color = "#ffffff"

        # Only append to tooltip path if this is not the root node
    if current_depth > 0 and parent_name is not None and threshold is not None and parent_values is not None and parent_samples is not None:
        condition_label = clean_node_label(parent_name, threshold, direction=direction)

        path = path + [(
            condition_label,
            int(total_samples[node_id]),  # number of samples at this node
            float(round(parent_gini, 3)) if parent_gini is not None else None,
            pass_count,
            fail_count
        )]

    # Leaf node or max depth
    if children_left[node_id] == _tree.TREE_LEAF or current_depth >= max_depth:
        class_id = int(np.argmax(values[node_id][0]))
        label = "Pass" if class_id == 1 else "Fail"

        label_text = clean_node_label(parent_name, threshold, direction=direction)

        # Final node stats
        total = int(total_samples[node_id])
        gini = round(tree.impurity[node_id], 3)
        pass_count = int(round(values[node_id][0][1] * sample_amount))
        fail_count = int(round(values[node_id][0][0] * sample_amount))

        final_label = f"{label_text}, ({total} samples of {parent_samples}), gini: {gini:.3f}, Pass: {pass_count}, Fail: {fail_count} → {label}"

        tooltip = format_ancestry_tooltip(path, final_label=final_label, init_samples_parent=total_samples[0])

        return {
            "name": f"{label_text}: {label}",
            "value": int(sample_amount),
            "tooltip": {"formatter": tooltip},
            "itemStyle": {
                "color": "#6effa6" if class_id == 1 else "#ff6a5a"
            }
        }


    current_node_name = "Node 0" if current_depth == 0 else clean_node_label(parent_name, threshold, direction=direction)

    left_id = children_left[node_id]
    right_id = children_right[node_id]

    left_samples = int(total_samples[left_id])
    right_samples = int(total_samples[right_id])
    total = left_samples + right_samples

    left_scaled = int(sample_amount * (left_samples / total)) if total > 0 else 0
    right_scaled = int(sample_amount * (right_samples / total)) if total > 0 else 0

    new_path = path  # already updated above
    current_gini = impurities[node_id]

    original_name = f"{feature_name} ≤ {threshold:.2f}"

    left_child = build_sunburst_json(
        tree, left_id, X_cols, max_depth, current_depth + 1,
        parent_name=original_name, direction = True,
        sample_amount=left_scaled, path=new_path, parent_gini=current_gini, parent_values=values[node_id][0], parent_samples=sample_amount
    )
    right_child = build_sunburst_json(
        tree, right_id, X_cols, max_depth, current_depth + 1,
        parent_name=original_name, direction = False,
        sample_amount=right_scaled, path=new_path, parent_gini=current_gini, parent_values=values[node_id][0], parent_samples=sample_amount
    )

    tooltip = format_ancestry_tooltip(new_path, init_samples_parent=total_samples[0])

    return {
        #"name": current_node_name + ", class:" + str(class_id) + ", pass:" + str(pass_count) + ", fail:" + str(fail_count) + " samples: " + str(sample_amount),
        "name": current_node_name,
        "value": int(left_child["value"] + right_child["value"]),
        "tooltip": {"formatter": tooltip},
        "itemStyle": {"color": color},
        "children": [left_child, right_child]
    }


def prune_layer_zero_and_shift_levels(node):
    """
    Removes the root node (layer 0) and promotes its children up one level.
    Essentially shifts layer 1 -> 0, 2 -> 1, etc.
    """
    if "children" in node:
        return node["children"] 
    return [node] 


def preprocess_bechdel_data(df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    df = df.copy()
    df['bechdel_pass'] = (df['BD_rating'] == 3).astype(int)

    numerical_cols = ['year', 'duration', 'avg_vote', 'votes',
                      'reviews_from_users', 'worlwide_gross_income', 'usa_gross_income']
    gender_cols = ['director_1_gender', 'writer_1_gender',
                   'director_2_gender', 'writer_2_gender',
                   'actor_1_gender', 'actor_2_gender']
    genre_cols = ['genre_1', 'genre_2']

    cols_to_use = list(set(selected_features + ['bechdel_pass']))
    df = df[cols_to_use + ['BD_rating']].copy()

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in genre_cols:
        if col in df.columns:
            df[col] = df[col].fillna("none")

    if 'language_1' in df.columns:
        df['language_1'] = df['language_1'].fillna("English")

    return df


def sunburst(filtered_data: pd.DataFrame, max_depth: int = 3, selected_features = all_features, min_samples_leaf : int = 30):

    df_clean = preprocess_bechdel_data(filtered_data, selected_features)

    clf, X, y = train_bechdel_tree(df_clean, selected_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    sunburst_data = build_sunburst_json(clf.tree_, 0, X.columns, max_depth=max_depth)

    prune_0 = prune_layer_zero_and_shift_levels(sunburst_data)

    option = {
        "title": {"text": ""},
        "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
        "series": [{
            "type": "sunburst",
            "data": prune_0,
            "radius": [0, "95%"],
            "sort": None,
            "highlightPolicy": "ancestor",
            "label": {
                "rotate": "tangential",
                "minAngle": 10,
                "overflow": "truncate",
                "borderRadius": 2,
                "color": "#000",                
                "textBorderColor": "#fff",  
                "textBorderWidth": 2 
            },
            "emphasis": {
                "focus": "ancestor",
                "label": {
                    "show": True,
                    "fontSize": 16,
                    "fontWeight": "bold",
                    "color": "#fff",
                    "backgroundColor": "rgba(0,0,0,0.7)",
                    "padding": 4,
                    "borderRadius": 4,
                    "textBorderColor": "#000",
                }
            },
        }]


    }

    st_echarts(options=option, height="1000px", width="100%")
    plot_bechdel_decision_tree(df_clean, selected_features, max_depth, min_samples_leaf=min_samples_leaf)

    #with st.expander("Sunburst Source Data (JSON)", expanded=False):
    #    st.json(prune_0)


def plot_bechdel_decision_tree(data: pd.DataFrame, selected_features: list = all_features, max_depth: int = 3, min_samples_leaf : int = 60):
    df = data.copy()

    clf, X, y = train_bechdel_tree(df, selected_features, max_depth=max_depth, min_samples_leaf = min_samples_leaf)

    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=X.columns,
        class_names=["Not Pass", "Pass"],
        filled=True,
        rounded=True,
        special_characters=True,
        node_ids=True
    )

    st.graphviz_chart(dot_data)
