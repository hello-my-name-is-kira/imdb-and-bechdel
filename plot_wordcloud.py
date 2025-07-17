import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_echarts import st_echarts
from collections import Counter
import re
import random


def wordcloud(data, max_words=200):

    text = " ".join(data['description'].dropna().astype(str))

    if not text.strip():
        st.warning("No description text available to generate a word cloud.")
        return

    wordcloud_img = WordCloud(
        width=1920,
        height=1080,
        background_color="#0e1117",
        colormap='plasma',
        max_words=max_words
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
    ax.imshow(wordcloud_img, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig)




def top_words_bar_chart(data, top_n=20):
    top_n = 20

    # Combine all descriptions
    text = " ".join(data['description'].dropna().astype(str).tolist()).lower()
    words = re.findall(r'\b\w+\b', text)

    # Filter out stopwords
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 2]

    # Count word frequencies
    word_counts = Counter(filtered_words)
    most_common = word_counts.most_common(top_n)

    if not most_common:
        st.warning("No significant words found after filtering.")
        return

    words, counts = zip(*most_common)
    words = list(words)[::-1]
    counts = list(counts)[::-1]

    # Generate N colors from a colormap
    cmap = plt.get_cmap("plasma") 
    colors = [cmap(i / (top_n - 1)) for i in range(top_n)]
    hex_colors = [f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})" for r, g, b, a in colors]

    # Build styled data
    styled_data = [{"value": count, "itemStyle": {"color": hex_colors[i]}} for i, count in enumerate(counts)]

    # Create ECharts horizontal bar chart
    option = {
        "title": {
            "text": f"Top {top_n} Words",
            "left": "center",
            "textStyle": {"color": "#fff"}
        },
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": "5%", "right": "5%", "bottom": "5%", "containLabel": True},
        "xAxis": {
            "type": "value",
            "axisLabel": {"color": "#aaa"},
            "axisLine": {"lineStyle": {"color": "#444"}}
        },
        "yAxis": {
            "type": "category",
            "data": words,
            "axisLabel": {"color": "#aaa"},
            "axisLine": {"lineStyle": {"color": "#444"}}
        },
        "series": [{
            "name": "Frequency",
            "type": "bar",
            "data": styled_data,

        }],
        "backgroundColor": "#11111100"
    }

    st_echarts(options=option, height="600px")


def bar_race_by_year(data, top_n=10):
    """
    Cumulative bar chart race of the most frequent words in movie descriptions by year.
    Each word is colored uniquely, and new words accumulate over time.
    """

    data = data.copy()
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year', 'description'])
    data['year'] = data['year'].astype(int)

    years = sorted(data['year'].unique())
    options = []
    timeline_labels = []
    color_map = {}
    cumulative_counter = Counter()

    def random_color():
        return f"rgb({random.randint(30, 220)}, {random.randint(30, 220)}, {random.randint(30, 220)})"

    for year in years:
        # Accumulate all previous + current year descriptions
        texts = data[data['year'] <= year]['description'].dropna().astype(str).str.lower()
        words = []
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text)
            words.extend([w for w in tokens if w not in STOPWORDS and len(w) > 2])

        cumulative_counter = Counter(words)
        word_counts = cumulative_counter.most_common(top_n)

        if not word_counts:
            continue

        labels, values = zip(*word_counts)
        labels = list(labels)[::-1]
        values = list(values)[::-1]

        # Assign color per word label
        styled_data = []
        for word, value in zip(labels, values):
            if word not in color_map:
                color_map[word] = random_color()
            styled_data.append({
                "value": value,
                "itemStyle": {"color": color_map[word]}
            })

        options.append({
            "title": {"text": f"Top {top_n} Words up to {year}", "textStyle": {"color": "#fff"}},
            "xAxis": {
                "type": "value",
                "axisLabel": {"color": "#ccc"},
                "axisLine": {"lineStyle": {"color": "#444"}}
            },
            "yAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {"color": "#ccc"},
                "axisLine": {"lineStyle": {"color": "#444"}}
            },
            "series": [{
                "type": "bar",
                "data": styled_data,
                "label": {"show": True, "position": "right", "color": "#fff"},
                "animationDurationUpdate": 2000,
                "animationEasing": "cubicOut"
            }],
            "grid": {"left": "5%", "right": "12%", "bottom": "5%", "containLabel": True},
            "backgroundColor": "#11111100"
        })

        timeline_labels.append(str(year))

    if not options:
        st.warning("No valid word data per year to animate.")
        return

    chart_option = {
        "baseOption": {
            "timeline": {
                "axisType": "category",
                "autoPlay": True,
                "playInterval": 2000,
                "right": "12%",
                "data": timeline_labels,
                "label": {"color": "#fff"}
            },
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "xAxis": {"type": "value"},
            "yAxis": {"type": "category"},
            "series": [{"type": "bar"}],
            "animationDuration": 2000,
            "animationEasing": "cubicOut"
        },
        "options": options
    }

    st_echarts(options=chart_option, height="600px")
