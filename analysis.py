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

    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    save_figs_path = os.getenv('save_figs')

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
        find_dotenv,
        load_dotenv,
        load_txt_files_to_dataframe,
        np,
        os,
        pd,
        plt,
        reshape_metrics,
        results_folder,
        save_figs_path,
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
def _(experiment_df, np, os, pd, plt, save_figs_path, sns):
    exp_median_cer = experiment_df[['dataset', 'deskew', 'max_ratio', 'cer_score']].groupby(['dataset', 'deskew', 'max_ratio'])['cer_score'].median().reset_index()

    exp_median_cer['Average'] = 'median'

    exp_mean_cer = experiment_df[['dataset', 'deskew', 'max_ratio', 'cer_score']].groupby(['dataset', 'deskew', 'max_ratio'])['cer_score'].mean().reset_index()

    exp_mean_cer['Average'] = 'mean'

    _plot_df = pd.concat([exp_median_cer, exp_mean_cer], ignore_index=True)

    _plot_df['max_ratio2'] = np.where(_plot_df['max_ratio']==1000, 'Inf', _plot_df['max_ratio'])

    g = sns.relplot(data = _plot_df, x = 'max_ratio2', y = 'cer_score', hue = 'dataset', style = 'deskew', 
                col = 'Average', kind = 'line')


    g.fig.suptitle("The Effect of Deskew and Image Cropping on CER scores", fontsize=16, y = 1.01)


    g.set_axis_labels('Maximum length to width ratio', "CER score")


    plt.savefig(os.path.join(save_figs_path,'deskew_crop_experiment.png'), 
                bbox_inches='tight', 
                dpi=300, 
                pad_inches=0.5)  # Adjust padding

    plt.show()
    return exp_mean_cer, exp_median_cer, g


@app.cell
def _(exp_mean_cer):
    exp_mean_cer.loc[exp_mean_cer['dataset'] != 'NCSE', ['cer_score', 'deskew', 'max_ratio']]
    return


@app.cell
def _():
    1-0.04/0.29
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
def _(mo):
    mo.md(
        r"""
        # Example image

        it is often nice to see an example of the algoritm in action. the below takes a specific image, and calculates the CER for each model. 

        The paper will then show the image and the CER Scores and the supplementary material will show the full texts
        """
    )
    return


@app.cell
def _(results_df):
    results_df.loc[results_df['file_name']=='NS2_1843-04-01_page_4_B0C5R42',['model', 'cer_score'] ].round(2)
    return


@app.cell
def _(results_df):
    results_df.loc[results_df['file_name'].isin(['EWJ_1859-03-01_page_17_B0C1R2', 'NS2_1843-04-01_page_4_B0C5R42']),
    ['file_name','model', 'cer_score'] ].round(2)
    return


@app.cell
def _(results_df):
    example_df = results_df.loc[results_df['file_name'].isin(['EWJ_1859-03-01_page_17_B0C1R2', 'NS2_1843-04-01_page_4_B0C5R42']),
    ['file_name','model', 'cer_score'] ].round(2)

    example_df['file_name'] = example_df['file_name'].str.extract('^([^_]*)')

    example_df
    return (example_df,)


@app.cell
def _(dataframe_to_latex_with_bold_extreme, example_df, reshape_metrics):
    example_table = reshape_metrics(example_df,  spread_col='file_name', agg_func='mean', round_digits=2).reset_index()

    example_table_latex = dataframe_to_latex_with_bold_extreme(example_table, extreme='min',  model_column='model', caption ='xxxx',
                                        label = 'tab:example_results')

    print(example_table)
    example_table_latex
    return example_table, example_table_latex


@app.cell
def _(results_df):
    output_contents = results_df.loc[results_df['file_name'].isin(
        ['EWJ_1859-03-01_page_17_B0C1R2', 'NS2_1843-04-01_page_4_B0C5R42']),['file_name','model','cer_score', 'content'] ]

    output_contents['file_name'] = output_contents['file_name'].str.extract('^([^_]*)')

    output_contents['content'] = output_contents['content'].str.replace('\n', ' ').str.replace('\r', ' ').str.replace('- ', '')

    output_contents['result'] = output_contents['content'].str[:200]
    output_contents.loc[output_contents['file_name']=='EWJ', ['model', 'result']]
    return (output_contents,)


@app.cell
def _(output_contents):
    output_contents.loc[output_contents['file_name']!='EWJ', ['model', 'result']]
    return


@app.cell
def _(output_contents):
    from tabulate import tabulate

    # Get the specific rows you want
    _table_data = output_contents.loc[output_contents['file_name']!='EWJ', ['model','cer_score', 'result']].round(2)

    # Create the LaTeX table manually
    _latex_table = (
        r'\begin{table}[h]' + '\n' +
        r'\caption{The first 200 characters of the Northern Star Example}' + '\n' +  # Add caption
        r'\label{tab:your_label_here}' + '\n' +      # Add label
        r'\small' + '\n' +
        r'\begin{tabular}{p{3cm}p{2cm}p{12cm}}' + '\n' +
        r'\hline' + '\n' +
        r'Model & CER & Result \\' + '\n' +
        r'\hline' + '\n'
    )

    # Add each row manually
    for _, _row in _table_data.iterrows():
        _latex_table += f"{_row['model']} & {_row['cer_score']} & {_row['result']} \\\\\n"

    _latex_table += (
        r'\hline' + '\n' +
        r'\end{tabular}' + '\n' +
        r'\end{table}'
    )

    print(_latex_table)
    return (tabulate,)


@app.cell
def _(output_contents):
    # Get the specific rows you want
    _table_data = output_contents.loc[output_contents['file_name']!='EWJ', ['model','cer_score', 'result']].round(2)

    # Create the LaTeX table manually
    _latex_table = (
        r'\begin{table}[h]' + '\n' +
        r'\small' + '\n' +
        r'\begin{tabular}{p{3cm}p{2cm}p{12cm}}' + '\n' +
        r'\hline' + '\n' +
        r'Model & CER & Result \\' + '\n' +
        r'\hline' + '\n'
    )

    # Add each row manually
    for _, _row in _table_data.iterrows():
        _latex_table += f"{_row['model']} & {_row['cer_score']} & {_row['result']} \\\\\n"

    _latex_table += (
        r'\hline' + '\n' +
        r'\end{tabular}' + '\n' +
        r'\end{table}'
    )

    print(_latex_table)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Performance of the paragraph matcher

        I 
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
