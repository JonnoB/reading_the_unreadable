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
    from analysis_functions import load_txt_files_to_dataframe, reshape_metrics,dataframe_to_latex_with_bold_extreme
    from jiwer import cer


    results_folder = "data/model_performance"

    BLN600_GT_path = "data/BLN600/Ground Truth"
    NCSE_GT_path = "data/transcripts/machine_response/"

    _BLN600_GT = load_txt_files_to_dataframe(BLN600_GT_path, 'GT')


    _NCSE_GT = load_txt_files_to_dataframe(NCSE_GT_path, 'GT')
    #
    #
    #
    # NEEEEEEEEDDDDDDDDDDDDSS TO BEEEEEEEEEE RTEPLACED WHEN GT READY!!!!!!!!!!!!!!!!!!!!
    #
    #
    GT_df = pd.concat([_BLN600_GT, _NCSE_GT])

    GT_df['file_name'] = GT_df['file_name'].str.replace("_box_page_id", "")
    return (
        BLN600_GT_path,
        GT_df,
        NCSE_GT_path,
        cer,
        dataframe_to_latex_with_bold_extreme,
        load_txt_files_to_dataframe,
        np,
        os,
        pd,
        reshape_metrics,
        results_folder,
    )


@app.cell
def _(GT_df, cer, load_txt_files_to_dataframe, os, pd, results_folder):
    results_df = []

    for _folder in os.listdir(results_folder):


        _temp = load_txt_files_to_dataframe(os.path.join(results_folder, _folder), 'content')
        _temp['folder'] = _folder
        results_df.append(_temp[['folder', 'file_name', 'content']])



    results_df = pd.concat(results_df, ignore_index=True).merge(GT_df, on ='file_name')
    results_df[['dataset', 'model']] = results_df['folder'].str.split('_', n=1, expand=True)

    results_df['cer_score'] = results_df.apply(lambda x: cer(x['GT'], x['content']), axis=1)
    return (results_df,)


@app.cell
def _(mo):
    mo.md(r"""The below result shows why you should not use mean when aggregating as the distributions are skewed. As can be seen the GOT mean CER is 1.41 whilst the median is 0.14. This discrepency is due GOT repeating the same phrase over and over, which appears to be a weakness of the VLM approach.""")
    return


@app.cell
def _(reshape_metrics, results_df):
    reshape_metrics(results_df,  spread_col='dataset', agg_func='mean', round_digits=2)
    return


@app.cell
def _(dataframe_to_latex_with_bold_extreme, reshape_metrics, results_df):
    median_table = reshape_metrics(results_df,  spread_col='dataset', agg_func='median', round_digits=2).reset_index()

    median_table_latex = dataframe_to_latex_with_bold_extreme(median_table, extreme='min',  model_column='model', caption ='The results show Pixtral outperforms all other models',
                                        label = 'tab:model_results')

    print(median_table_latex)
    median_table
    return median_table, median_table_latex


@app.cell
def _(reshape_metrics, results_df):
    reshape_metrics(results_df,  spread_col='dataset', agg_func='std', round_digits=2)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
