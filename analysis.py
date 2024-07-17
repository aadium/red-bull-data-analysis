import pandas as pd
import json


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

# Convert the dictionary to a JSON string
json_data = json.dumps(data_dict, indent=4)

# Write the JSON data to a file
with open('task_scores.json', 'w') as file:
    file.write(json_data)
