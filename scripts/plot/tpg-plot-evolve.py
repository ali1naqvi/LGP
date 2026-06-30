#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import csv
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_pdf import PdfPages

'''
The script is included within the environmental variables for TPG.

A comprehensive guide can be found in the link below.

https://gitlab.cas.mcmaster.ca/kellys32/tpg/-/wikis/TPG-Generation-Plot-for-CSV-Logging-Files.

To use, simply call the script inside any experiments within `experiment_directories/*`.
    
    :param csv_files (Optional): The name of csv file(s) to plot
    :param column_name (Required): The name of the column to plot against generations 
    :return: Returns nothing but saves a .png file for the plot

For the optional parameter `csv_files`, possible values can be:
    
    - 'all-selection' | 'all-removal' | 'all-timing' | 'all-replacement' (retrieves all CSV files within the directory)
    - 'csv_file1.csv,csv_file2.csv, ...' (can have one or more specific CSV files listed with same type, separated by comma `,`)
    - 'csv_file1,csv_file2, ...' (similar to above, but doesn't need the ".csv" extension in the file name)
    - left blank (the program is setup to default to 'all-selection')

Example:
    
    tpg-plot-evolve.py all-selection best_fitness
    tpg-plot-evolve.py selection.42.42.csv,selection.42.43.csv best_fitness
    tpg-plot-evolve.py timing.42.42,timing.42.43 generation_time
    tpg-plot-evolve.py best_fitness
'''

log_dir = "logs"
plot_dir = "plots"

# Values near DBL_MAX (e.g., crowding distance boundary points) can overflow
# Matplotlib's autoscaling/padding. Treat these as missing for plotting.
INF_LIKE_THRESHOLD = 1e100

# When plotting validation metrics that are only updated every N generations,
# treat each N-generation update as "one iteration" on the x-axis.
DEFAULT_ITERATION_MOD = 50

# Self-modifying programs expose mutation probabilities in working memory while
# inheriting starting probabilities from evolved constants. Keep those elite
# columns together so each mutation operation can be read as output vs. start.
MUTATION_RATE_START_PAIRS = {
    "elite_avg_output_rate_swap": "elite_avg_start_rate_swap",
    "elite_avg_output_rate_delete": "elite_avg_start_rate_delete",
    "elite_avg_output_rate_add": "elite_avg_start_rate_add",
    "elite_avg_output_rate_mutate": "elite_avg_start_rate_mutate",
    "elite_avg_output_rate_s5": "elite_avg_start_rate_s5",
}
START_TO_MUTATION_RATE = {
    start: output for output, start in MUTATION_RATE_START_PAIRS.items()
}

SELF_MODIFYING_PROBABILITY_EPSILON = 1e-6
SELF_MODIFYING_RATE_TEMPERATURE = 1.0

def stable_sigmoid(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value >= 0.0:
        z = np.exp(-value)
        return 1.0 / (1.0 + z)
    z = np.exp(value)
    return z / (1.0 + z)

def raw_tendency_to_probability(value: float) -> float:
    """Match SelfModifyingRawTendencyToProbability in misc.h."""
    return (
        SELF_MODIFYING_PROBABILITY_EPSILON
        + (1.0 - 2.0 * SELF_MODIFYING_PROBABILITY_EPSILON)
        * stable_sigmoid(value / SELF_MODIFYING_RATE_TEMPERATURE)
    )

def mutation_rate_probability_series(series: pd.Series) -> pd.Series:
    """Convert logged raw tendencies to probabilities when needed."""
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values
    # New logs store probabilities in (epsilon, 1-epsilon). Older logs stored raw
    # tendencies or clamped them to [0, 1] without sigmoid.
    if finite.min() < -0.01 or finite.max() > 1.01:
        return values.map(
            lambda v: raw_tendency_to_probability(v) if np.isfinite(v) else np.nan
        )
    return values

def get_unique_filename(base_filename):
    """
    Generates a unique filename by appending an increment if the file already exists.
    
    :param base_filename: The base filename (e.g., 'output.png')
    :return: A unique filename (e.g., 'output_1.png', 'output_2.png', etc.)
    """
    filename, ext = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename
    
    while os.path.exists(f"{plot_dir}/{new_filename}"):
        new_filename = f"{filename}_{counter}{ext}"
        counter += 1
    
    return new_filename

def get_csv_columns(filepath):
    """Extracts column names from a CSV file."""
    with open(filepath, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        column_names = next(reader)
        return column_names[1:]

def capitalize_snake_case(s):
    return ' '.join(word.capitalize() for word in s.split('_'))

def mutation_rate_plot_columns(column_name):
    """Return the output and evolved-start columns for a paired plot."""
    output_column = START_TO_MUTATION_RATE.get(column_name, column_name)
    start_column = MUTATION_RATE_START_PAIRS.get(output_column)
    return output_column, start_column

def plot_generations_single(csv_files, column_name, pdf = None):
    """
    Plots the given csv_files and column name against generations.
    
    :param csv_files: The name of csv file(s) to plot (e.g., 'tma.42.12345.csv')
    :param column_name: The column to plot against 'generation' (e.g., 'best_fitness')
    :return: None, but saves the output into a .png file
    """
    output_column, start_column = mutation_rate_plot_columns(column_name)
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('tab10') # a list of 10 color schemes
    line_styles = ['-', '--', '-.', ':']
    
    valid_files = []
    x_label = "Generation"
    
    # start processing the listed csv files
    for idx, csv_file in enumerate(csv_files):
        try:
            # Use full path to read the CSV file
            full_path = os.path.join(log_dir, csv_file)
            df = pd.read_csv(full_path)
            if 'generation' not in df.columns:
                print(f"Skipping {csv_file}: missing 'generation' column")
                continue
            if output_column not in df.columns:
                print(f"Skipping {csv_file}: missing '{output_column}' column")
                continue
            

            # Mutation-rate pairs use fixed colours to make output vs. start
            # immediately readable. Other metrics retain the
            # per-file colour/style scheme for combined plots.
            is_mutation_rate_pair = start_column is not None
            color = "tab:blue" if is_mutation_rate_pair else cmap(idx % 10)
            line_style = '-' if is_mutation_rate_pair else line_styles[(idx // 10) % len(line_styles)]
            label = os.path.splitext(os.path.basename(csv_file))[0]
            
            x = pd.to_numeric(df['generation'], errors='coerce')
            y = (
                mutation_rate_probability_series(df[output_column])
                if is_mutation_rate_pair
                else pd.to_numeric(df[output_column], errors='coerce')
            )

            # If we're plotting validation_fitness, only show the generations where
            # validation actually ran (value != 0), and plot against "iteration"
            # where 1 iteration == DEFAULT_ITERATION_MOD generations.
            if output_column == "validation_fitness":
                nonzero = y.fillna(0.0) != 0.0
                x = x[nonzero]
                y = y[nonzero]
                x = x / float(DEFAULT_ITERATION_MOD)
                x_label = f"Iteration (every {DEFAULT_ITERATION_MOD} generations)"

            bad = (~np.isfinite(y)) | (y.abs() >= INF_LIKE_THRESHOLD)
            if bad.any():
                y = y.mask(bad, np.nan)

            if y.notna().sum() == 0:
                print(f"Skipping {csv_file}: '{output_column}' contains only inf-like/NaN values")
                continue

            has_start = start_column in df.columns
            is_elite_average = output_column.startswith("elite_avg_")
            series_name = "Elite-average output" if is_elite_average else "Output value"
            output_label = series_name if len(csv_files) == 1 else f"{label} — {series_name.lower()}"
            plt.plot(x, y,
                    marker='o' if len(csv_files) == 1 else '',
                    linestyle=line_style,
                    color=color,
                    label=output_label,
                    alpha=0.8)

            if has_start:
                start_y = mutation_rate_probability_series(df[start_column])
                start_bad = (~np.isfinite(start_y)) | (start_y.abs() >= INF_LIKE_THRESHOLD)
                start_y = start_y.mask(start_bad, np.nan)
                if start_y.notna().sum() > 0:
                    start_name = (
                        "Elite-average start" if is_elite_average
                        else "Evolved start"
                    )
                    start_label = (
                        start_name if len(csv_files) == 1
                        else f"{label} — {start_name.lower()}"
                    )
                    plt.plot(x, start_y,
                             marker='s' if len(csv_files) == 1 else '',
                             linestyle='-',
                             color='tab:red',
                             label=start_label,
                             alpha=0.8)
                else:
                    print(f"Skipping evolved start values for {csv_file}: '{start_column}' contains only inf-like/NaN values")
            elif start_column:
                print(f"Note: {csv_file} has no '{start_column}' column; plotting output values only")
            
            # necessary to keep track whether files are being processed
            valid_files.append(csv_file)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

    if not valid_files:
        print("No valid CSV files with required columns found!")
        return

    # configure properties of the graph (x-axis, y-axis, grid)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(capitalize_snake_case(output_column), fontsize=12)

    plt.grid(True, alpha=0.3)
    
    # Paired mutation-rate plots always need a legend; other plots only need
    # one when they combine several CSV files.
    if len(valid_files) > 1 or start_column:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if pdf:
        pdf.savefig()
    else:
        output_filename = f"{output_column}_vs_generations"

        # multiple file plots will end with '_combined' 
        if len(valid_files) > 1:
            output_filename += "_combined"
        output_filename += ".pdf"

        # if output file already exists, add a number in the end
        output_filename =  f"{plot_dir}/{get_unique_filename(output_filename)}"

        plt.savefig(output_filename)
        print(f"Plot saved to '{output_filename}'")

    plt.close()

def plot_generations_multiple(csv_files, column_names):
    # Get prefix from the first file - use basename to handle subdirectory structure
    first_file = os.path.basename(csv_files[0]) if csv_files else ""
    prefix = first_file.split('.')[0] if first_file else ""

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    output_filename = plot_dir + "/" + get_unique_filename(f"{prefix}_all_vs_generations.pdf")

    with PdfPages(output_filename) as pdf:
        # Start-rate columns are rendered alongside their corresponding output
        # rate rather than as redundant standalone figures.
        start_columns = set(MUTATION_RATE_START_PAIRS.values())
        for column_name in column_names:
            if column_name in start_columns:
                continue
            plot_generations_single(csv_files, column_name, pdf)

    print(f"All plots saved to '{output_filename}'")

if __name__ == "__main__": 
    Path("plots").mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description='Plot CSV column against generations')

    # csv_files defaults to 'all-selection' if left blank, plotting all CSV within the directory
    parser.add_argument('csv_files', type=str, nargs="?", default='all-selection', help='Comma-separated list of CSV files (or use "all" for *.csv)')
    parser.add_argument('column_name', type=str, help='Column name to plot against generations')
    
    args = parser.parse_args()
    
    # handle "all-*" keyword or comma-separated files
    prefixes = {"selection", "removal", "timing", "replacement"}
    csv_key = args.csv_files.lower()

    if csv_key.startswith("all-"):
        prefix = csv_key[4:]  # extract part after "all-"
        if prefix in prefixes:
            # Use direct path construction - don't rely on chdir
            prefix_dir = os.path.join(log_dir, prefix)
            
            if not os.path.exists(prefix_dir):
                raise ValueError(f"Subdirectory '{prefix}' not found in {log_dir}")
            
            csv_files = [f for f in os.listdir(prefix_dir) if f.endswith('.csv')]
            if not csv_files:
                print(f"No CSV files found in {prefix_dir}")
                sys.exit(1)
                
            # Store paths relative to log_dir
            csv_files = [os.path.join(prefix, f) for f in csv_files]
        else:
            raise ValueError(f"Invalid prefix '{prefix}'. Expected one of {prefixes}.")
    else:
        # Extract files and determine their prefixes
        raw_files = [f.strip() for f in args.csv_files.split(',')]
        csv_files = []
        
        for f in raw_files:
            if not f.lower().endswith(".csv"):
                f = f"{f}.csv"
                
            # Extract prefix from filename
            prefix = f.split('.')[0].lower()
            if prefix not in prefixes:
                raise ValueError(f"File '{f}' doesn't have a valid prefix. Expected one of {prefixes}.")
            
            # Store with subdirectory
            csv_files.append(os.path.join(prefix, f))

        csv_files = list(set(csv_files))  # remove duplicates
    
    valid_files = []
    for f in csv_files:
        full_path = os.path.join(log_dir, f)
        if not os.path.exists(full_path):
            print(f"Warning: File '{full_path}' not found")
        else:
            valid_files.append(f)
    
    if not valid_files:
        print("No valid CSV files found!")
        sys.exit(1)

    column_name = args.column_name

    try:
        if column_name == "all":
            # Get columns from the first valid file using full path
            full_path = os.path.join(log_dir, valid_files[0])
            column_names = get_csv_columns(full_path)
            plot_generations_multiple(valid_files, column_names)
        else:
            plot_generations_single(valid_files, column_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
