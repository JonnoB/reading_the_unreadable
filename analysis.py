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
    import seaborn as sns
    import matplotlib.pyplot as plt

    results_folder = "data/model_performance"

    BLN600_GT_path = "data/BLN600/Ground Truth"
    NCSE_GT_path = "data/transcripts/machine_response/"

    data_folder = "ffff"

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
        data_folder,
        dataframe_to_latex_with_bold_extreme,
        load_txt_files_to_dataframe,
        np,
        os,
        pd,
        plt,
        reshape_metrics,
        results_folder,
        sns,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Deskew and Crop experiment.

        This section tests how best to use the pixtral model and it's ability to handle text boxes of various elongations.

        The results of this are the basis for parametrising the pre-processing in the rest of the work.
        """
    )
    return


@app.cell
def _(GT_df, cer, os, pd):
    experiment_folder = "data/download_jobs/experiments/dataframe"
    experiment_csvs = os.listdir(experiment_folder)
    experiment_df = []

    for _file in experiment_csvs:
        # Read the CSV file
        _temp = pd.read_csv(os.path.join(experiment_folder, _file))

        # Parse the filename
        # Split by underscore and remove the .csv extension
        parts = _file.replace('.csv', '').split('_')

        # Extract the components
        dataset = parts[0]  # BLN600 or NCSE
        deskew = parts[2]   # True or False
        max_ratio = float(parts[5])  # 1.5 or 1000

        # Add new columns
        _temp['file'] = _file
        _temp['dataset'] = dataset
        _temp['deskew'] = deskew
        _temp['max_ratio'] = max_ratio
        if dataset == 'BLN600':
            _temp['file_name'] = (_temp['filename']
                                 .str.replace("_box_page_id", "", regex=False)
                                 .str.replace("_page_1_B0C1R0", "", regex=False)
                                 .str.replace(".txt", "", regex=False))
        else:  # NCSE
            _temp['file_name'] = (_temp['filename']
                                 .str.replace("_box_page_id", "", regex=False)
                                 .str.replace(".txt", "", regex=False))



        # Append to list of dataframes
        experiment_df.append(_temp)

    # Combine all dataframes
    experiment_df = pd.concat(experiment_df, ignore_index=True).merge(GT_df, left_on ='file_name', right_on ='file_name')

    experiment_df['cer_score'] = experiment_df.apply(lambda x: cer(x['GT'], x['content']), axis=1)
    return (
        dataset,
        deskew,
        experiment_csvs,
        experiment_df,
        experiment_folder,
        max_ratio,
        parts,
    )


@app.cell
def _(experiment_df, np, plt, sns):
    exp_median_cer = experiment_df[['dataset', 'deskew', 'max_ratio', 'cer_score']].groupby(['dataset', 'deskew', 'max_ratio'])['cer_score'].median().reset_index()
    plt.title("Comparison of Deskew and\nmaximum allowed width to length ratio")


    exp_median_cer['max_ratio2'] = np.where(exp_median_cer['max_ratio']==1000, 'Inf', exp_median_cer['max_ratio'])

    sns.lineplot(data = exp_median_cer, x = 'max_ratio2', y = 'cer_score', hue = 'dataset', style = 'deskew')
    plt.xlabel ('Maximum length to width ratio')

    plt.ylabel("Median CER score")
    return (exp_median_cer,)


@app.cell
def _(exp_median_cer):
    exp_median_cer.loc[exp_median_cer['dataset'] == 'NCSE', ['cer_score', 'deskew', 'max_ratio']].describe()
    return


@app.cell
def _(experiment_df):
    experiment_df[['dataset', 'deskew', 'max_ratio', 'cer_score']].groupby(['dataset', 'deskew', 'max_ratio'])['cer_score'].describe()
    return


@app.cell
def _(experiment_df, plt, sns):
    sns.catplot(data = experiment_df[['dataset', 'deskew', 'max_ratio', 'cer_score']], x = 'max_ratio', y = 'cer_score',
                kind = 'box',
               hue = 'dataset',
    col = 'deskew')
    plt.ylim([0, 0.1])

    plt.show()
    return


@app.cell
def _(experiment_df):
    experiment_df2 = experiment_df
    experiment_df2['model'] = experiment_df2['file']
    return (experiment_df2,)


@app.cell
def _(experiment_df2, reshape_metrics):
    reshape_metrics(experiment_df2,  spread_col='dataset', agg_func='mean', round_digits=2)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Create the model comparison

        The below section is the main event when it comes to proving the approach. It loads the OCR text from the 5 different models and finds the CER score for them.
        """
    )
    return


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
