import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t


def get_data():
    return pd.read_csv('./data.csv')  # Update path if needed


def calculate_95_confidence_intervals(data):
    # Define z-score for 95% confidence interval
    z_score = 1.96

    # Create a dictionary to store confidence intervals grouped by factor
    grouped_conf_intervals = {}

    for _, row in data.iterrows():
        ci_dict = {
            'task': row['task'],
            'measure': row['measure'],
            'placebo CI': (round(row['placebo mean'] - z_score * row['placebo error'], 10),
                           round(row['placebo mean'] + z_score * row['placebo error'], 10)),
            'rbs CI': (round(row['rbs mean'] - z_score * row['rbs error'], 10),
                       round(row['rbs mean'] + z_score * row['rbs error'], 10)),
            'rb CI': (round(row['rb mean'] - z_score * row['rb error'], 10),
                      round(row['rb mean'] + z_score * row['rb error'], 10))
        }

        factor = row['factor score']
        if factor not in grouped_conf_intervals:
            grouped_conf_intervals[factor] = []
        grouped_conf_intervals[factor].append(ci_dict)

    return grouped_conf_intervals


def calculate_p_values(data, sample_size):
    # Create a dictionary to store p-values grouped by factor
    grouped_p_values = {}

    for _, row in data.iterrows():
        # Calculate t-statistic and p-value for placebo
        t_stat_placebo = row['placebo mean'] / (row['placebo error'] / (sample_size ** 0.5))
        p_value_placebo = t.sf(abs(t_stat_placebo), df=sample_size - 1) * 2

        # Calculate t-statistic and p-value for rbs
        t_stat_rbs = row['rbs mean'] / (row['rbs error'] / (sample_size ** 0.5))
        p_value_rbs = t.sf(abs(t_stat_rbs), df=sample_size - 1) * 2

        # Calculate t-statistic and p-value for rb
        t_stat_rb = row['rb mean'] / (row['rb error'] / (sample_size ** 0.5))
        p_value_rb = t.sf(abs(t_stat_rb), df=sample_size - 1) * 2

        p_values_dict = {
            'task': row['task'],
            'measure': row['measure'],
            't_stats': {
                'placebo': t_stat_placebo,
                'rbs': t_stat_rbs,
                'rb': t_stat_rb
            },
            'p_values': {
                'placebo': p_value_placebo,
                'rbs': p_value_rbs,
                'rb': p_value_rb
            },
        }

        factor = row['factor score']
        if factor not in grouped_p_values:
            grouped_p_values[factor] = []
        grouped_p_values[factor].append(p_values_dict)

    return grouped_p_values


def draw_bar_chart_means(data):
    # Flatten the means for plotting
    means_flat = []
    for _, row in data.iterrows():
        for drink in ['placebo', 'rbs', 'rb']:
            means_flat.append({
                'factor': row['factor score'],
                'drink': drink,
                'mean': row[f'{drink} mean']
            })

    # Convert to DataFrame
    means_df = pd.DataFrame(means_flat)

    # Get unique factor scores
    unique_factors = means_df['factor'].unique()

    # Plot bar chart for each factor score
    for factor in unique_factors:
        factor_df = means_df[means_df['factor'] == factor]
        plt.figure(figsize=(12, 6))
        sns.barplot(x='drink', y='mean', data=factor_df, ci=None)
        plt.title(f'Bar Chart of Means for Factor {factor}')
        plt.savefig(f'means_chart_factor_{factor}.png')
        plt.show()


raw_data = get_data()

# Calculate confidence intervals
confidence_intervals = calculate_95_confidence_intervals(raw_data)
p_values = calculate_p_values(raw_data, sample_size=24)

# Save p-values to a JSON file
with open('p_values.json', 'w') as f:
    json.dump(p_values, f, indent=4)

# Save confidence intervals to a JSON file
with open('confidence_intervals.json', 'w') as f:
    json.dump(confidence_intervals, f, indent=4)

# Draw bar chart with error bars for means and errors
draw_bar_chart_means(raw_data)
