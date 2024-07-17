import pandas as pd
import json

from scipy.stats import ttest_ind_from_stats


def get_data():
    return pd.read_csv('./data.csv')  # Update path if needed


def categorize_factor_score(cat, data):
    return data[data['factor score'] == cat]


def get_task_names(data):
    return list(data['task'])


def get_task_scores(data, item_task, score_type):
    # Filter data by task
    filtered_data = data[data['task'] == item_task]

    # Determine columns to select based on score_type
    if score_type == 'mean':
        columns_to_select = ['measure', 'placebo mean', 'rbs mean', 'rb mean']
    elif score_type == 'error':
        columns_to_select = ['measure', 'placebo error', 'rbs error', 'rb error']
    else:
        return "Invalid score_type"

    # Construct a list of dictionaries for each row in the filtered data
    scores_list = []
    for _, row in filtered_data.iterrows():
        scores_dict = {col: row[col] for col in columns_to_select}
        scores_list.append(scores_dict)

    return scores_list


def calculate_95_confidence_intervals(data):
    # Define z-score for 95% confidence interval
    z_score = 1.96

    # Create a dictionary to store confidence intervals grouped by factor
    grouped_conf_intervals = {}

    for _, row in data.iterrows():
        ci_dict = {
            'task': row['task'],
            'measure': row['measure'],
            'placebo CI': (round(row['placebo mean'] - z_score * row['placebo error'], 3),
                           round(row['placebo mean'] + z_score * row['placebo error'], 3)),
            'rbs CI': (round(row['rbs mean'] - z_score * row['rbs error'], 3),
                       round(row['rbs mean'] + z_score * row['rbs error'], 3)),
            'rb CI': (round(row['rb mean'] - z_score * row['rb error'], 3),
                      round(row['rb mean'] + z_score * row['rb error'], 3))
        }

        factor = row['factor score']
        if factor not in grouped_conf_intervals:
            grouped_conf_intervals[factor] = []
        grouped_conf_intervals[factor].append(ci_dict)

    return grouped_conf_intervals


def calculate_p_values(data, sample_size=30):
    # Create a dictionary to store p-values grouped by factor
    grouped_p_values = {}

    for _, row in data.iterrows():
        # Perform independent two-sample t-test between placebo and RBS
        t_statistic, p_value_rbs = ttest_ind_from_stats(
            mean1=row['placebo mean'], std1=row['placebo error'], nobs1=sample_size,
            mean2=row['rbs mean'], std2=row['rbs error'], nobs2=sample_size,
            equal_var=False
        )

        # Perform independent two-sample t-test between placebo and RB
        t_statistic, p_value_rb = ttest_ind_from_stats(
            mean1=row['placebo mean'], std1=row['placebo error'], nobs1=sample_size,
            mean2=row['rb mean'], std2=row['rb error'], nobs2=sample_size,
            equal_var=False
        )

        p_value_dict = {
            'task': row['task'],
            'measure': row['measure'],
            'p-value (placebo vs RBS)': p_value_rbs,
            'p-value (placebo vs RB)': p_value_rb
        }

        factor = row['factor score']
        if factor not in grouped_p_values:
            grouped_p_values[factor] = []
        grouped_p_values[factor].append(p_value_dict)

    return grouped_p_values


raw_data = get_data()
attentional_intensity_index_data = categorize_factor_score('attentional intensity index', raw_data)
sustained_attention_index_data = categorize_factor_score('sustained attention index', raw_data)
memory_capacity_index_data = categorize_factor_score('memory capacity index', raw_data)
speed_of_retrival_index_data = categorize_factor_score('speed of retrival index', raw_data)

attentional_intensity_index_tasks = get_task_names(attentional_intensity_index_data)
sustained_attention_index_tasks = get_task_names(sustained_attention_index_data)
memory_capacity_index_tasks = get_task_names(memory_capacity_index_data)
speed_of_retrival_index_tasks = get_task_names(speed_of_retrival_index_data)

# Calculate confidence intervals
confidence_intervals = calculate_95_confidence_intervals(raw_data)
p_values = calculate_p_values(raw_data, sample_size=22)

# Save p-values to a JSON file
with open('p_values.json', 'w') as f:
    json.dump(p_values, f, indent=4)

# Save confidence intervals to a JSON file
with open('confidence_intervals.json', 'w') as f:
    json.dump(confidence_intervals, f, indent=4)
