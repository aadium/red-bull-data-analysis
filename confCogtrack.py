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


raw_data = get_data()

# Calculate confidence intervals
confidence_intervals = calculate_95_confidence_intervals(raw_data)

# Save confidence intervals to a JSON file
with open('confidence_intervals.json', 'w') as f:
    json.dump(confidence_intervals, f, indent=4)
