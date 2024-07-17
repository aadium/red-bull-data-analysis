import pandas as pd
import json
from scipy.stats import ttest_ind_from_stats


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


def calculate_p_values(data, sample_size=30):
    p_values_grouped = {}

    for _, row in data.iterrows():
        factor_score = row['factor score']
        task = row['task']
        measure = row['measure']

        # Means and standard errors
        means = [row['placebo mean'], row['rbs mean'], row['rb mean']]
        errors = [row['placebo error'], row['rbs error'], row['rb error']]

        # Calculate t-tests
        t_stat1, p_val1 = ttest_ind_from_stats(mean1=means[0], std1=errors[0], nobs1=sample_size,
                                               mean2=means[1], std2=errors[1], nobs2=sample_size)
        t_stat2, p_val2 = ttest_ind_from_stats(mean1=means[0], std1=errors[0], nobs1=sample_size,
                                               mean2=means[2], std2=errors[2], nobs2=sample_size)
        t_stat3, p_val3 = ttest_ind_from_stats(mean1=means[1], std1=errors[1], nobs1=sample_size,
                                               mean2=means[2], std2=errors[2], nobs2=sample_size)

        p_values_dict = {
            'task': task,
            'measure': measure,
            'placebo vs rbs': round(p_val1, 10),
            'placebo vs rb': round(p_val2, 10),
            'rbs vs rb': round(p_val3, 10)
        }

        if factor_score not in p_values_grouped:
            p_values_grouped[factor_score] = []
        p_values_grouped[factor_score].append(p_values_dict)

    return p_values_grouped


raw_data = get_data()

# Calculate confidence intervals
confidence_intervals = calculate_95_confidence_intervals(raw_data)
p_values = calculate_p_values(raw_data, sample_size=22)

# Save p-values to a JSON file
with open('p_values.json', 'w') as f:
    json.dump(p_values, f, indent=4)

# Save confidence intervals to a JSON file
with open('confidence_intervals.json', 'w') as f:
    json.dump(confidence_intervals, f, indent=4)
