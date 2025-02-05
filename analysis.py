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
    from function_modules.analysis_functions import load_txt_files_to_dataframe, reshape_metrics,dataframe_to_latex_with_bold_extreme, clean_text
    from jiwer import cer
    import seaborn as sns
    import matplotlib.pyplot as plt

    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    save_figs_path = os.getenv('save_figs')

    results_folder = "data/model_performance"

    BLN600_GT_path = "data/BLN600/Ground Truth"
    NCSE_GT_path = "data/transcripts/ground_truth/"

    data_folder = "ffff"

    _BLN600_GT = load_txt_files_to_dataframe(BLN600_GT_path, 'GT')


    _NCSE_GT = load_txt_files_to_dataframe(NCSE_GT_path, 'GT')

    GT_df = pd.concat([_BLN600_GT, _NCSE_GT])

    GT_df['file_name'] = GT_df['file_name'].str.replace("_box_page_id", "")
    return (
        BLN600_GT_path,
        GT_df,
        NCSE_GT_path,
        cer,
        clean_text,
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
        # Bounding-Box poat processing quality analysis

        By measuring the overlap and coverage of the bounding boxes produced by the two post-processing methods relative to the bounding-boxes producced by DocLayout-Yolo we can evaluate which post-processing if any should be chosen. The results show that in generall the two post-processing methods perform similarly, both generally reducing overlap or leaving it unchanged and increasing coverage by the same amount when controlling for periodical. However, in the case of the Northern Star the column filling approach provides a an increase of almost 10 percentage points over the simpler post-processing method and 24 percentage points in comparison to no post-processing. This increase takes the total median coverage to 100\% compared to 92\% and 79\% for simple post-processing and no-post processing respectively. 
        The Northern Star behaves differently to the other periodicals due to the size of the page and that it has 4 or 5 columns which is much more than the other papers.

        As a result, all the papers will use the simple post-processing without column filling apart from the Northern Star which will use column filling.
        """
    )
    return


@app.cell
def _(os, pd):
    save_folder = "data/overlap_coverage"
    source_folders = ["post_process", 'post_process_fill', "post_process_raw"]

    # List to store individual dataframes
    dfs = []

    for method in source_folders:
        method_folder = os.path.join(save_folder, method)

        # Process each file in the method folder
        for file in os.listdir(method_folder):

            # Load the parquet file
            df = pd.read_parquet(os.path.join(method_folder, file))

            # Add method column
            df['method'] = method
            df['periodical'] = file

            dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    _df = combined_df.loc[combined_df['method']=="post_process_raw", ['page_id','perc_print_area_overlap', 'perc_print_area_coverage']]
    combined_df = combined_df.merge(_df, on = 'page_id', suffixes = ['', "_raw"])
    combined_df['overlap_diff'] = combined_df['perc_print_area_overlap'] - combined_df['perc_print_area_overlap_raw']
    combined_df['coverage_diff'] = combined_df['perc_print_area_coverage'] - combined_df['perc_print_area_coverage_raw']

    combined_df['p'] = combined_df['periodical'].apply(lambda x: x.split("_")[0])

    combined_df.loc[combined_df['method']!='post_process_raw'].groupby(['p', 'method']).agg(
        overlap_mean=('overlap_diff', 'mean'),
        overlap_median=('overlap_diff', 'median'),
        coverage_mean=('coverage_diff', 'mean'),
        coverage_median=('coverage_diff', 'median')
    ).round(2)
    return (
        combined_df,
        df,
        dfs,
        file,
        method,
        method_folder,
        save_folder,
        source_folders,
    )


@app.cell
def _(combined_df):
    combined_df.groupby([ 'p', 'method'])['perc_print_area_overlap'].agg(['mean', 'median']).round(2)

    combined_df.loc[:].groupby(['p', 'method']).agg(
        overlap_mean=('perc_print_area_overlap', 'mean'),
        overlap_median=('perc_print_area_overlap', 'median'),
        coverage_mean=('perc_print_area_coverage', 'mean'),
        coverage_median=('perc_print_area_coverage', 'median')
    ).round(2)
    return


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
def _(GT_df, cer, clean_text, os, pd):
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

    #clean line breaks and set to lower
    experiment_df['content'] = experiment_df['content'].apply(clean_text)
    experiment_df['content'] = experiment_df['content'].str.lower()

    experiment_df['GT'] = experiment_df['GT'].apply(clean_text)
    experiment_df['GT'] = experiment_df['GT'].str.lower()

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


    #_plot_df = _plot_df.loc[_plot_df['max_ratio2']!='1.5']

    g = sns.FacetGrid(data=_plot_df, col='Average', row = 'dataset' ,sharey=False)
    g.map_dataframe(sns.lineplot, x='max_ratio2', y='cer_score', hue='deskew')

    g.fig.suptitle("The Effect of Deskew and Image Cropping on CER scores", fontsize=16, y=1.01)
    g.set_axis_labels('Maximum length to width ratio', "CER score")

    # Add legend (since we're using a different construction method)
    g.add_legend()

    plt.savefig(os.path.join(save_figs_path,'deskew_crop_experiment.png'), 
                bbox_inches='tight', 
                dpi=300, 
                pad_inches=0.5)

    plt.show()
    return exp_mean_cer, exp_median_cer, g


@app.cell
def _(experiment_df):
    experiment_df.groupby(['dataset', 'max_ratio'])['total_tokens'].agg(['mean', 'median'])
    return


@app.cell
def _(experiment_df):
    experiment_df[['dataset', 'deskew', 'max_ratio', 'cer_score']].groupby(['dataset', 'deskew', 'max_ratio'])['cer_score'].agg(['mean', 'median'])
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
def _(
    GT_df,
    cer,
    clean_text,
    load_txt_files_to_dataframe,
    os,
    pd,
    results_folder,
):
    results_df = []
    print(os.listdir(results_folder))
    for _folder in os.listdir(results_folder):


        _temp = load_txt_files_to_dataframe(os.path.join(results_folder, _folder), 'content')
        _temp['folder'] = _folder
        results_df.append(_temp[['folder', 'file_name', 'content']])



    results_df = pd.concat(results_df, ignore_index=True).merge(GT_df, on ='file_name')
    results_df[['dataset', 'model']] = results_df['folder'].str.split('_', n=1, expand=True)

    #clean line breaks and set to lower
    results_df['content'] = results_df['content'].apply(clean_text)
    results_df['content'] = results_df['content'].str.lower()

    results_df['GT'] = results_df['GT'].apply(clean_text)
    results_df['GT'] = results_df['GT'].str.lower()

    results_df['cer_score'] = results_df.apply(lambda x: cer(x['GT'], x['content']), axis=1)

    results_df['no_error'] = results_df['cer_score']==0
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
def _(plt, results_df, sns):
    pivot_df = results_df.loc[results_df['model'].isin(['pixtral', 'pixtral_large']) ].pivot(
        index='file_name',
        columns='model',
        values='cer_score'
    )

    # Create the scatterplot
    sns.scatterplot(data=pivot_df, x='pixtral', y='pixtral_large')
    plt.show()
    pivot_df.describe()
    return (pivot_df,)


@app.cell
def _(pivot_df):
    thresholds = [10, 1, 0.2, 0.1, 0.01]

    for threshold in thresholds:
        pixtral_fraction = (pivot_df['pixtral'] < threshold).mean()
        pixtral_large_fraction = (pivot_df['pixtral_large'] < threshold).mean()
        
        print(f"\nThreshold {threshold}:")
        print(f"Pixtral: {pixtral_fraction:.3%} below threshold")
        print(f"Pixtral Large: {pixtral_large_fraction:.3%} below threshold")
    return pixtral_fraction, pixtral_large_fraction, threshold, thresholds


@app.cell
def _(mo):
    mo.md(
        r"""
        # Example image

        it is often nice to see an example of the algoritm in action. the below takes a specific image, and calculates the CER for each model. 

        The paper will then show the image and the CER Scores and the supplementary material will show the full texts


        This is a very interesting image, because there is a scratch through it 'NS2_1843-04-01_page_4_B0C5R42'. GOT and tesseract do very well even outperforming Pixtral, however, it would be confusing to show this as it doesn't represent overall performance. As such I will use 'NS2_1843-04-01_page_4_B0C1R1' as this is pretty close to the mean for all and at least in the correct order.
        """
    )
    return


@app.cell
def _(results_df):
    # Find an example that is broadly representative of the average scores
    results_df2 = results_df.copy()

    results_df2['rank'] = results_df2.groupby('file_name')['cer_score'].rank(method='min')
    results_df2['med_score'] = results_df2.groupby('file_name')['cer_score'].transform('median')
    results_df2['diff_score'] =results_df2['med_score'] - results_df2['cer_score']


    results_df2.loc[(results_df2['model']=='pixtral') & (results_df2['rank']==1) & (~results_df2['folder'].str.contains('BLN')) & (results_df2['cer_score']<0.08)].sort_values('diff_score')
    return (results_df2,)


@app.cell
def _(results_df):
    results_df.loc[results_df['file_name']=='NS2_1843-04-01_page_4_B0C1R1',['model', 'cer_score'] ].round(2)
    return


@app.cell
def _(results_df):
    output_contents = results_df.loc[results_df['file_name'].isin(
        ['NS2_1843-04-01_page_4_B0C1R1']),['file_name','model','cer_score', 'content'] ]

    output_contents['file_name'] = output_contents['file_name'].str.extract('^([^_]*)')

    output_contents['content'] = output_contents['content'].str.replace('\n', ' ').str.replace('\r', ' ').str.replace('- ', '')

    output_contents['result'] = output_contents['content'].str[:200]
    output_contents.loc[:, ['model', 'result']]
    return (output_contents,)


@app.cell
def _(output_contents):
    from tabulate import tabulate

    # Get the specific rows you want
    _table_data = output_contents.loc[:, ['model','cer_score', 'result']].round(2).sort_values('cer_score')

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
def _(results_df):
    results_df.loc[results_df['model']=='pixtral'].sort_values('cer_score')
    return


@app.cell
def _(results_df):
    results_df.groupby('model')['no_error'].sum()*100/len(results_df['file_name'].unique())
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
