import pandas as pd
import json
from scipy.stats import t, ttest_ind_from_stats


def get_data():
    return pd.read_csv('data.csv')


def categorize_factor_score(cat, data):
    return data[data['factor score'] == cat]


def get_task_names(data):
    return list(data['task'])


def get_task_scores(data, task, score_type):
    # Filter data by task
    filtered_data = data[data['task'] == task]

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


def calculate_confidence_intervals(data):
    data = data.copy()

    # Calculate the differences in means
    data['diff_rbs_placebo'] = data['rbs mean'] - data['placebo mean']
    data['diff_rb_placebo'] = data['rb mean'] - data['placebo mean']
    data['diff_rbs_rb'] = data['rbs mean'] - data['rb mean']

    # Calculate the standard errors for the differences
    data['se_diff_rbs_placebo'] = (data['rbs error']**2 + data['placebo error']**2)**0.5
    data['se_diff_rb_placebo'] = (data['rb error']**2 + data['placebo error']**2)**0.5
    data['se_diff_rbs_rb'] = (data['rbs error']**2 + data['rb error']**2)**0.5

    # Define the confidence level
    confidence = 0.95
    alpha = 1 - confidence

    # Calculate the t-critical value
    degrees_of_freedom = len(data) - 1
    t_critical = t.ppf(1 - alpha/2, degrees_of_freedom)

    # Calculate the confidence intervals
    data['ci_diff_rbs_placebo'] = t_critical * data['se_diff_rbs_placebo']
    data['ci_diff_rb_placebo'] = t_critical * data['se_diff_rb_placebo']
    data['ci_diff_rbs_rb'] = t_critical * data['se_diff_rbs_rb']

    # Perform t-tests and add p-values to the DataFrame
    data['p_value_rbs_placebo'] = data.apply(lambda row: ttest_ind_from_stats(
        mean1=row['rbs mean'], std1=row['rbs error'], nobs1=30,
        mean2=row['placebo mean'], std2=row['placebo error'], nobs2=30)[1], axis=1)

    data['p_value_rb_placebo'] = data.apply(lambda row: ttest_ind_from_stats(
        mean1=row['rb mean'], std1=row['rb error'], nobs1=30,
        mean2=row['placebo mean'], std2=row['placebo error'], nobs2=30)[1], axis=1)

    data['p_value_rbs_rb'] = data.apply(lambda row: ttest_ind_from_stats(
        mean1=row['rbs mean'], std1=row['rbs error'], nobs1=30,
        mean2=row['rb mean'], std2=row['rb error'], nobs2=30)[1], axis=1)

    p_values = data[['p_value_rbs_placebo', 'p_value_rb_placebo', 'p_value_rbs_rb']]
    confidence_intervals = data[['ci_diff_rbs_placebo', 'ci_diff_rb_placebo', 'ci_diff_rbs_rb']]

    return p_values, confidence_intervals


raw_data = get_data()
attentional_intensity_index_data = categorize_factor_score('attentional intensity index', raw_data)
sustained_attention_index_data = categorize_factor_score('sustained attention index', raw_data)
memory_capacity_index_data = categorize_factor_score('memory capacity index', raw_data)
speed_of_retrival_index_data = categorize_factor_score('speed of retrival index', raw_data)

attentional_intensity_index_tasks = get_task_names(attentional_intensity_index_data)
sustained_attention_index_tasks = get_task_names(sustained_attention_index_data)
memory_capacity_index_tasks = get_task_names(memory_capacity_index_data)
speed_of_retrival_index_tasks = get_task_names(speed_of_retrival_index_data)

# Initialize a dictionary to hold all the data
data_dict = {
    'Attentional Intensity Index Tasks': {},
    'Sustained Attention Index Tasks': {},
    'Memory Capacity Index Tasks': {},
    'Speed of Retrieval Index Tasks': {}
}

# Populate the dictionary with data
for task in attentional_intensity_index_tasks:
    data_dict['Attentional Intensity Index Tasks'][task] = {
        'mean': get_task_scores(attentional_intensity_index_data, task, 'mean'),
        'error': get_task_scores(attentional_intensity_index_data, task, 'error')
    }

for task in sustained_attention_index_tasks:
    data_dict['Sustained Attention Index Tasks'][task] = {
        'mean': get_task_scores(sustained_attention_index_data, task, 'mean'),
        'error': get_task_scores(sustained_attention_index_data, task, 'error')
    }

for task in memory_capacity_index_tasks:
    data_dict['Memory Capacity Index Tasks'][task] = {
        'mean': get_task_scores(memory_capacity_index_data, task, 'mean'),
        'error': get_task_scores(memory_capacity_index_data, task, 'error')
    }

for task in speed_of_retrival_index_tasks:
    data_dict['Speed of Retrieval Index Tasks'][task] = {
        'mean': get_task_scores(speed_of_retrival_index_data, task, 'mean'),
        'error': get_task_scores(speed_of_retrival_index_data, task, 'error')
    }

# Print the dictionary
# print(json.dumps(data_dict, indent=4))

# Calculate confidence intervals
p_values, confidence_intervals = calculate_confidence_intervals(attentional_intensity_index_data)
print(p_values)
print(confidence_intervals)