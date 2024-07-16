import pandas as pd


def get_data():
    return pd.read_csv('data.csv')


def categorize_factor_score(cat, data):
    return data[data['factor score'] == cat]


def get_attentional_intensity_index_scores(data):
    return categorize_factor_score('attentional intensity index', data)


def get_sustained_attention_index_scores(data):
    return categorize_factor_score('sustained attention index', data)


def get_memory_capacity_index_scores(data):
    return categorize_factor_score('memory capacity index', data)


def get_speed_of_retrival_index_scores(data):
    return categorize_factor_score('speed of retrival index', data)


raw_data = get_data()
attentional_intensity_index_scores = get_attentional_intensity_index_scores(raw_data)
sustained_attention_index_scores = get_sustained_attention_index_scores(raw_data)
memory_capacity_index_scores = get_memory_capacity_index_scores(raw_data)
speed_of_retrival_index_scores = get_speed_of_retrival_index_scores(raw_data)

print(attentional_intensity_index_scores)
print()
print(sustained_attention_index_scores)
print()
print(memory_capacity_index_scores)
print()
print(speed_of_retrival_index_scores)
