import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    import os 


    data = os.path.join('data', "download_jobs/EWJ.parquet")

    data_path = os.path.join('data', "download_jobs", "ncse")

    all_data_files = [file for file in os.listdir(data_path) if '.parquet' in file ]
    return Path, all_data_files, data, data_path, os, pd


@app.cell
def _(all_data_files, data_path, os, pd):

    test2 = []

    for _file in all_data_files:

        test2.append(pd.read_parquet(os.path.join(data_path, _file)))

    test2 = pd.concat(test2, ignore_index=True)
    return (test2,)


@app.cell
def _(test2):
    test2.groupby('class').size()
    return


@app.cell
def _(test2):
    test2
    return


if __name__ == "__main__":
    app.run()
