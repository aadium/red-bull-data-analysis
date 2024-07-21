import numpy as np
import scipy.stats as stats
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def filter_and_format_data(data, factor_score):
    filtered_data = data[data['factor score'] == factor_score]
    formatted_data = {}

    for _, row in filtered_data.iterrows():
        task_measure = f"{row['task']} - {row['measure']}"
        formatted_data[task_measure] = {
            'placebo': {'mean': row['placebo mean'], 'se': row['placebo error']},
            'rbsf': {'mean': row['rbs mean'], 'se': row['rbs error']},
            'rb': {'mean': row['rb mean'], 'se': row['rb error']}
        }

    print(factor_score.capitalize())
    return formatted_data

# Load data
file_path = './data.csv'  # Update path if needed
data = filter_and_format_data(load_data(file_path), factor_score='speed of retrival index')

# Sample sizes
n_placebo = 24
n_rbsf = 24
n_rb = 24

# Function to calculate t-value
def calculate_t(mean1, mean2, se1, se2, n1, n2):
    se_diff = np.sqrt(se1 ** 2 + se2 ** 2)
    t_value = (mean1 - mean2) / se_diff
    df = n1 + n2 - 2
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df))
    return t_value, p_value

# Function to calculate Cohen's d
def calculate_cohens_d(mean1, mean2, se1, se2, n1, n2):
    # Convert standard errors to standard deviations
    sd1 = se1 * np.sqrt(n1)
    sd2 = se2 * np.sqrt(n2)

    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    d_value = (mean1 - mean2) / pooled_sd
    return d_value

# Function to calculate and store t, p, and d values for all conditions and drinks
def calculate_statistics(data, n_placebo, n_rbsf, n_rb):
    results = {
        "Metric": ["t", "p", "d"]
    }

    for condition, drinks in data.items():
        for drink, values in drinks.items():
            mean = values['mean']
            se = values['se']
            t_value = 0
            p_value = 0
            d_value = 0
            if drink == 'placebo':
                t_value, p_value = calculate_t(mean, 0, se, 0, n_placebo, n_placebo)
                d_value = calculate_cohens_d(mean, 0, se, 0, n_placebo, n_placebo)
            elif drink == 'rbsf':
                t_value, p_value = calculate_t(mean, 0, se, 0, n_rbsf, n_rbsf)
                d_value = calculate_cohens_d(mean, 0, se, 0, n_rbsf, n_rbsf)
            elif drink == 'rb':
                t_value, p_value = calculate_t(mean, 0, se, 0, n_rb, n_rb)
                d_value = calculate_cohens_d(mean, 0, se, 0, n_rb, n_rb)

            key = f"{condition} - {drink}"
            results[key] = [t_value, p_value, d_value]

    return results

# Calculate and store statistics
results = calculate_statistics(data, n_placebo, n_rbsf, n_rb)

# Create a DataFrame from the results dictionary
df = pd.DataFrame(results)

# Save the DataFrame to a CSV file
df.to_csv('result.csv', index=False)