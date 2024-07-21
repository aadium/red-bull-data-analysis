import numpy as np
import scipy.stats as stats

# Graph data
mean_placebo_hr = -4.2
se_placebo_hr = 1
mean_rbsf_hr = -2.6
se_rbsf_hr = 1
mean_rb_hr = -4.8
se_rb_hr = 1

mean_placebo_sbp = 1.6
se_placebo_sbp = 1.3
mean_rbsf_sbp = 3.6
se_rbsf_sbp = 1.2
mean_rb_sbp = 4.4
se_rb_sbp = 1.25

mean_placebo_dbp = 5
se_placebo_dbp = 1
mean_rbsf_dbp = 6
se_rbsf_dbp = 1.1
mean_rb_dbp = 1.6
se_rb_dbp = 1.1

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
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

    # Calculate Cohen's d
    d_value = (mean1 - mean2) / pooled_sd
    return d_value


# Calculate t, p, and d values for 'hr' item
t_hr_placebo, p_hr_placebo = calculate_t(mean_placebo_hr, 0, se_placebo_hr, 0, n_placebo,
                                         n_placebo)
d_hr_placebo = calculate_cohens_d(mean_placebo_hr, 0, se_placebo_hr, 0, n_placebo, n_placebo)

t_hr_rbsf, p_hr_rbsf = calculate_t(mean_rbsf_hr, 0, se_rbsf_hr, 0, n_rbsf, n_rbsf)
d_hr_rbsf = calculate_cohens_d(mean_rbsf_hr, 0, se_rbsf_hr, 0, n_rbsf, n_rbsf)

t_hr_rb, p_hr_rb = calculate_t(mean_rb_hr, 0, se_rb_hr, 0, n_rb, n_rb)
d_hr_rb = calculate_cohens_d(mean_rb_hr, 0, se_rb_hr, 0, n_rb, n_rb)

# Calculate t, p, and d values for 'sbp' item
t_sbp_placebo, p_sbp_placebo = calculate_t(mean_placebo_sbp, 0, se_placebo_sbp, 0, n_placebo,
                                           n_placebo)
d_sbp_placebo = calculate_cohens_d(mean_placebo_sbp, 0, se_placebo_sbp, 0, n_placebo, n_placebo)

t_sbp_rbsf, p_sbp_rbsf = calculate_t(mean_rbsf_sbp, 0, se_rbsf_sbp, 0, n_rbsf, n_rbsf)
d_sbp_rbsf = calculate_cohens_d(mean_rbsf_sbp, 0, se_rbsf_sbp, 0, n_rbsf, n_rbsf)

t_sbp_rb, p_sbp_rb = calculate_t(mean_rb_sbp, 0, se_rb_sbp, 0, n_rb, n_rb)
d_sbp_rb = calculate_cohens_d(mean_rb_sbp, 0, se_rb_sbp, 0, n_rb, n_rb)

# Calculate t, p, and d values for 'dbp' item
t_dbp_placebo, p_dbp_placebo = calculate_t(mean_placebo_dbp, 0, se_placebo_dbp, 0, n_placebo,
                                           n_placebo)
d_dbp_placebo = calculate_cohens_d(mean_placebo_dbp, 0, se_placebo_dbp, 0, n_placebo, n_placebo)

t_dbp_rbsf, p_dbp_rbsf = calculate_t(mean_rbsf_dbp, 0, se_rbsf_dbp, 0, n_rbsf, n_rbsf)
d_dbp_rbsf = calculate_cohens_d(mean_rbsf_dbp, 0, se_rbsf_dbp, 0, n_rbsf, n_rbsf)

t_dbp_rb, p_dbp_rb = calculate_t(mean_rb_dbp, 0, se_rb_dbp, 0, n_rb, n_rb)
d_dbp_rb = calculate_cohens_d(mean_rb_dbp, 0, se_rb_dbp, 0, n_rb, n_rb)

# Print results
print(f"Heart Rate (Placebo): t={t_hr_placebo}, p={p_hr_placebo}, d={d_hr_placebo}")
print(f"Heart Rate (RBSF): t={t_hr_rbsf}, p={p_hr_rbsf}, d={d_hr_rbsf}")
print(f"Heart Rate (RB): t={t_hr_rb}, p={p_hr_rb}, d={d_hr_rb}")

print(f"Systolic BP (Placebo): t={t_sbp_placebo}, p={p_sbp_placebo}, d={d_sbp_placebo}")
print(f"Systolic BP (RBSF): t={t_sbp_rbsf}, p={p_sbp_rbsf}, d={d_sbp_rbsf}")
print(f"Systolic BP (RB): t={t_sbp_rb}, p={p_sbp_rb}, d={d_sbp_rb}")

print(f"Diastolic BP (Placebo): t={t_dbp_placebo}, p={p_dbp_placebo}, d={d_dbp_placebo}")
print(f"Diastolic BP (RBSF): t={t_dbp_rbsf}, p={p_dbp_rbsf}, d={d_dbp_rbsf}")
print(f"Diastolic BP (RB): t={t_dbp_rb}, p={p_dbp_rb}, d={d_dbp_rb}")
