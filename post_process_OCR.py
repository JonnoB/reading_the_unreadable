import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from pathlib import Path
    import os 


    data = os.path.join('data', "download_jobs/EWJ.parquet")
    return Path, data, os, pd


@app.cell
def _(pd):
    def split_paragraphs(df):
        # Create a list to store the new rows
        new_rows = []
        
        # Iterate through the original dataframe
        for idx, row in df.iterrows():
            # Split the content by \n\n
            paragraphs = row['content'].split('\n\n')
            
            # Create new rows for each paragraph
            for i, para in enumerate(paragraphs, 1):
                # Create a copy of the original row
                new_row = row.copy()
                # Update the content with just this paragraph
                new_row['content'] = para.strip()
                # Add paragraph number
                new_row['paragraph_number'] = i
                new_rows.append(new_row)
        
        # Create new dataframe from the list of rows
        return pd.DataFrame(new_rows).reset_index(drop=True)

    def calculate_paragraph_reading_order(df):
        """
        Calculate paragraph-level reading order within each page of each issue.
        
        Parameters:
        df: DataFrame with columns 'issue_id', 'page_number', 'reading_order', 'paragraph_number'
        
        Returns:
        DataFrame with new 'reading_order_para' column
        """
        # Sort the dataframe to ensure correct order
        df_sorted = df.sort_values(['issue_id', 'page_number', 'reading_order', 'paragraph_number'])
        
        # Group by issue and page, then calculate cumulative count
        df_sorted['reading_order_para'] = df_sorted.groupby(['issue_id', 'page_number']).cumcount() + 1
        
        return df_sorted
    return calculate_paragraph_reading_order, split_paragraphs


@app.cell
def _(calculate_paragraph_reading_order, data, pd, split_paragraphs):
    test = pd.read_parquet(data)

    test = split_paragraphs(test)

    test = calculate_paragraph_reading_order(test)
    #Remove line breaks within the paragraphs
    test['content'] = test['content'].str.replace("-\n", "").str.replace("\n", " ")
    return (test,)


@app.cell
def _(test):
    test
    return


@app.cell
def _(test):
    test['content'].iloc[5]
    return


if __name__ == "__main__":
    app.run()
