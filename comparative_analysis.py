import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import os
    from gliner import GLiNER
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    import textstat

    save_figs = os.environ["save_figs"]

    text_type_path = "data/classification/text_type_classified"
    IPTC_topic_path = "data/classification/IPTC_type_classified"

    processed_bboxes_path = "data/download_jobs/ncse/dataframes/post_processed"

    text_code_dict = {0: "article", 1: "advert", 2: "poem/song/story", 3: "other"}

    iptc_mapping = {
        0: "arts_culture_entertainment_media",
        1: "crime_law_justice",
        2: "disaster_accident_emergency",
        3: "economy_business_finance",
        4: "education",
        5: "environment",
        6: "health",
        7: "human_interest",
        8: "labour",
        9: "lifestyle_leisure",
        10: "politics",
        11: "religion",
        12: "science_technology",
        13: "society",
        14: "sport",
        15: "conflict_war_peace",
        16: "weather",
        17: "NA",
    }
    return (
        GLiNER,
        IPTC_topic_path,
        cosine_similarity,
        iptc_mapping,
        np,
        os,
        pd,
        plt,
        processed_bboxes_path,
        save_figs,
        sch,
        sns,
        squareform,
        text_code_dict,
        text_type_path,
        textstat,
    )


@app.cell
def _(
    IPTC_topic_path,
    iptc_mapping,
    os,
    pd,
    processed_bboxes_path,
    text_type_path,
):
    # Define paths for saved files
    data_folder = "data"
    sampled_topics_file = os.path.join(data_folder, "sampled_text_topics.parquet")
    topic_distribs_file = os.path.join(data_folder, "topic_distribs.parquet")

    # Create data folder if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Check if files already exist
    if os.path.exists(sampled_topics_file) and os.path.exists(topic_distribs_file):
        print("Loading existing files...")
        sampled_text_topics = pd.read_parquet(sampled_topics_file)
        topic_distribs = pd.read_parquet(topic_distribs_file)
    else:
        print("Creating new files...")
        sampled_text_topics = []
        topic_distribs = []

        for _target_file in os.listdir(processed_bboxes_path):
            print(_target_file)
            _temp = pd.read_parquet(os.path.join(processed_bboxes_path, _target_file))
            _temp["bbox_uid"] = _temp["box_page_id"] + "_" + _temp["page_id"]
            _temp = _temp.loc[_temp["class"] == "text"]

            _temp = _temp[["bbox_uid", "class"]]

            _text_type_sample = pd.read_parquet(
                os.path.join(text_type_path, "classified_" + _target_file)
            )
            _text_type_sample = _text_type_sample.loc[
                (_text_type_sample["predicted_class"] == 0)
                & (_text_type_sample["bbox_uid"].isin(_temp["bbox_uid"]))
            ]

            del _temp
            _text_type_sample = _text_type_sample.sample(2000, random_state=1891)

            _IPTC_type_sample = pd.read_parquet(
                os.path.join(IPTC_topic_path, "classified_" + _target_file)
            )

            _sum_IPTC = _IPTC_type_sample.drop(columns="bbox_uid").sum().to_frame().T

            _sum_IPTC["periodical"] = _target_file

            _IPTC_type_sample = _IPTC_type_sample.filter(regex="probability|bbox_uid")

            _IPTC_type_sample = _IPTC_type_sample.loc[
                _IPTC_type_sample["bbox_uid"].isin(_text_type_sample["bbox_uid"])
            ]

            _IPTC_type_sample["periodical"] = _target_file

            sampled_text_topics.append(_IPTC_type_sample)

            topic_distribs.append(_sum_IPTC)

            del _IPTC_type_sample

        sampled_text_topics = pd.concat(sampled_text_topics, ignore_index=True)
        topic_distribs = pd.concat(topic_distribs, ignore_index=True)

        # Clean up periodical names
        for df in [sampled_text_topics, topic_distribs]:
            df["periodical"] = df["periodical"].str.replace("_PDF_files.parquet", "")
            df["periodical"] = df["periodical"].str.replace("_issue", "")
            df["periodical"] = df["periodical"].str.replace("_", " ")

        topic_distribs = topic_distribs.loc[
            :, ~topic_distribs.columns.str.endswith("probability")
        ]

        # Save the created files
        sampled_text_topics.to_parquet(sampled_topics_file)
        topic_distribs.to_parquet(topic_distribs_file)

    topic_distribs[topic_distribs.select_dtypes("number").columns] = (
        topic_distribs.select_dtypes(
            "number"
        ).div(topic_distribs.select_dtypes("number").sum(axis=1), axis=0)
    )

    # Create new mapping for the actual column names
    column_mapping = {f"class_{k}": v for k, v in iptc_mapping.items()}

    # Rename the columns
    topic_distribs = topic_distribs.rename(columns=column_mapping)
    return (
        column_mapping,
        data_folder,
        df,
        sampled_text_topics,
        sampled_topics_file,
        topic_distribs,
        topic_distribs_file,
    )


@app.cell
def _(plt, sns, topic_distribs):
    # First melt the dataframe to get it into the right format for seaborn
    melted_df = topic_distribs.melt(
        id_vars=["periodical"], var_name="category", value_name="proportion"
    )

    melted_df["category"] = melted_df["category"].str.replace(
        "arts_culture_entertainment_media", "Arts/Entertainment"
    )
    melted_df["category"] = melted_df["category"].str.replace(
        "economy_business_finance", "Business/Finance"
    )
    melted_df["category"] = melted_df["category"].str.replace("_", " ")
    melted_df["category"] = melted_df["category"].str.title()

    g = sns.catplot(
        data=melted_df,
        x="category",
        y="proportion",
        col="periodical",
        kind="bar",
        col_wrap=3,
        height=3,
        aspect=1.5,
        legend=True,
    )

    # Rotate labels for each subplot
    for ax in g.axes.flat:
        plt.sca(ax)
        plt.xticks(rotation=45, ha="right")

    # Adjust the layout to prevent label cutoff
    plt.tight_layout()

    plt.show()
    return ax, g, melted_df


@app.cell
def _(melted_df):
    summary_df = (
        melted_df.groupby("periodical")
        .apply(lambda x: x.nlargest(3, "proportion"))
        .reset_index(drop=True)
    )

    summary_df = (
        summary_df.groupby("periodical")["category"]
        .agg(lambda x: ", ".join(x))
        .reset_index()
    )

    summary_df = summary_df.rename(
        columns={"category": "Top Categories", "periodical": "Periodicals"}
    )
    summary_df
    return (summary_df,)


@app.cell
def _(summary_df):
    latex_table = summary_df.to_latex(index=False, escape=False)

    # Print the LaTeX code
    print(latex_table)
    return (latex_table,)


@app.cell
def _(melted_df):
    bottom_3_per_periodical = (
        melted_df.groupby("periodical")
        .apply(lambda x: x.nsmallest(3, "proportion"))
        .reset_index(drop=True)
    )

    _summary_df = (
        bottom_3_per_periodical.groupby("periodical")["category"]
        .agg(lambda x: ", ".join(x))
        .reset_index()
    )

    _summary_df
    return (bottom_3_per_periodical,)


@app.cell
def _(
    cosine_similarity,
    np,
    os,
    plt,
    save_figs,
    sch,
    squareform,
    topic_distribs,
):
    # Calculate cosine similarity matrix (if not already calculated)
    numerical_cols = topic_distribs.drop("periodical", axis=1)
    similarity_matrix = cosine_similarity(numerical_cols)

    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(squareform(distance_matrix), method="ward")

    # Create dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(
        linkage_matrix,
        labels=topic_distribs["periodical"].values,
        leaf_rotation=90,  # rotates the x axis labels
        leaf_font_size=10,  # font size for the x axis labels
    )
    plt.xticks(rotation=25, ha="right")
    plt.title(
        "Hierarchical Clustering Dendrogram showing topic cosine\nsimilarity between the periodicals"
    )
    plt.xlabel("Periodicals")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_figs, "hierarchical_similarity.png"))
    plt.show()
    return (
        dendrogram,
        distance_matrix,
        linkage_matrix,
        numerical_cols,
        similarity_matrix,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Reading ease

        Comparing the reading ease between the periodicals
        """
    )
    return


@app.cell
def _(os, pd, processed_bboxes_path, sampled_text_topics, textstat):
    bbox_uid_list = sampled_text_topics["bbox_uid"].tolist()  # Convert to list

    reading_ease_df = []

    for _file in os.listdir(processed_bboxes_path):
        _temp = pd.read_parquet(os.path.join(processed_bboxes_path, _file))
        _temp["bbox_uid"] = _temp["box_page_id"] + "_" + _temp["page_id"]
        _temp = _temp.loc[_temp["bbox_uid"].isin(bbox_uid_list)]
        _temp["reading_ease"] = _temp["content"].apply(textstat.flesch_reading_ease)
        _temp = _temp[["bbox_uid", "reading_ease"]]
        _temp["periodical"] = _file
        reading_ease_df.append(_temp)

    reading_ease_df = pd.concat(reading_ease_df)

    reading_ease_df["periodical"] = reading_ease_df["periodical"].str.replace(
        "_PDF_files.parquet", ""
    )
    reading_ease_df["periodical"] = reading_ease_df["periodical"].str.replace(
        "_issue", ""
    )
    reading_ease_df["periodical"] = reading_ease_df["periodical"].str.replace("_", " ")
    return bbox_uid_list, reading_ease_df


@app.cell
def _(os, plt, reading_ease_df, save_figs, sns):
    plt.clf()
    sns.boxplot(
        data=reading_ease_df.loc[reading_ease_df["reading_ease"] > 0].sort_values(
            "periodical"
        ),
        x="periodical",
        y="reading_ease",
    )
    plt.xticks(rotation=25, ha="right")
    plt.title("Flesch-Kincaid readability by periodical")
    plt.tight_layout()
    plt.savefig(os.path.join(save_figs, "readability.png"))
    plt.show()
    return


@app.cell
def _(reading_ease_df):
    reading_ease_df.groupby("periodical")["reading_ease"].describe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # NER popularity comparision


        Looking at the popularity of Crystal Palace/ Great exhibition and of Palmeston. comparing the Northern Star and the Leader
        """
    )
    return


@app.cell
def _(os, pd, processed_bboxes_path):
    _file = os.listdir(processed_bboxes_path)[0]

    temporal_comparison = []

    for _file in [
        "Leader_issue_PDF_files.parquet",
        "Northern_Star_issue_PDF_files.parquet",
    ]:
        _temp = pd.read_parquet(os.path.join(processed_bboxes_path, _file))

        _temp = _temp.loc[_temp["issue_id"].str.contains("1851")]

        _temp = _temp[
            _temp["content"].str.contains("crystal palace|great exhibition", case=False)
        ]
        _temp["periodical"] = _file
        temporal_comparison.append(_temp)

        del _temp

    temporal_comparison = pd.concat(temporal_comparison, ignore_index=True)

    temporal_comparison["date"] = (
        temporal_comparison["issue_id"].str.split("-", n=1).str[1]
    )

    temporal_comparison["date"] = pd.to_datetime(temporal_comparison["date"])

    temporal_comparison["periodical"] = temporal_comparison["periodical"].str.replace(
        "_PDF_files.parquet", ""
    )
    temporal_comparison["periodical"] = temporal_comparison["periodical"].str.replace(
        "_issue", ""
    )
    temporal_comparison["periodical"] = temporal_comparison["periodical"].str.replace(
        "_", " "
    )
    return (temporal_comparison,)


@app.cell
def _(os, pd, plt, save_figs, sns, temporal_comparison):
    # First, ensure 'date' is in datetime format
    temporal_comparison["date"] = pd.to_datetime(temporal_comparison["date"])

    # Group by periodical and date, get size
    temp = (
        temporal_comparison.groupby(["periodical", "date"])
        .size()
        .reset_index(name="count")
    )

    # Set the date as index
    temp = temp.set_index("date")

    # Group by periodical and resample weekly
    temp = temp.groupby("periodical")["count"].resample("ME").sum()

    # Reset the index
    temp = temp.reset_index()

    # Convert date to month names
    temp["month"] = temp["date"].dt.strftime("%B")

    # Create the plot
    sns.lineplot(data=temp, x="month", y="count", hue="periodical")

    # Optionally, improve the formatting
    plt.xticks(rotation=25, ha="right")
    plt.title("Coverage of the Great Exhibition at Crystal Palace by month")
    plt.xlabel("Month")
    plt.ylabel("Count")

    plt.savefig(os.path.join(save_figs, "crystal_palace.png"))

    plt.show()
    return (temp,)


@app.cell
def _(temp):
    temp.groupby("periodical")["count"].sum()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
