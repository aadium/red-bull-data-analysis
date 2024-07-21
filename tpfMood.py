import numpy as np
import scipy.stats as stats

# Graph data
mean_placebo_alert = 10
se_placebo_alert = 3.22
mean_rbsf_alert = 10
se_rbsf_alert = 3.22
mean_rb_alert = 22
se_rb_alert = 3.22

mean_placebo_tired = -6
se_placebo_tired = 3.5
mean_rbsf_tired = -14
se_rbsf_tired = 3.5
mean_rb_tired = -15.5
se_rb_tired = 3.5

mean_placebo_jittery = 6.5
se_placebo_jittery = 3.5
mean_rbsf_jittery = 13.3
se_rbsf_jittery = 3.4
mean_rb_jittery = 13.2
se_rb_jittery = 3.4

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


# Calculate t, p, and d values for 'alert' item
t_alert_placebo, p_alert_placebo = calculate_t(mean_placebo_alert, 0, se_placebo_alert, 0, n_placebo,
                                               n_placebo)
d_alert_placebo = calculate_cohens_d(mean_placebo_alert, 0, se_placebo_alert, 0, n_placebo, n_placebo)

t_alert_rbsf, p_alert_rbsf = calculate_t(mean_rbsf_alert, 0, se_rbsf_alert, 0, n_rbsf, n_rbsf)
d_alert_rbsf = calculate_cohens_d(mean_rbsf_alert, 0, se_rbsf_alert, 0, n_rbsf, n_rbsf)

t_alert_rb, p_alert_rb = calculate_t(mean_rb_alert, 0, se_rb_alert, 0, n_rb, n_rb)
d_alert_rb = calculate_cohens_d(mean_rb_alert, 0, se_rb_alert, 0, n_rb, n_rb)

# Calculate t, p, and d values for 'tired' item
t_tired_placebo, p_tired_placebo = calculate_t(mean_placebo_tired, 0, se_placebo_tired, 0, n_placebo,
                                               n_placebo)
d_tired_placebo = calculate_cohens_d(mean_placebo_tired, 0, se_placebo_tired, 0, n_placebo, n_placebo)

t_tired_rbsf, p_tired_rbsf = calculate_t(mean_rbsf_tired, 0, se_rbsf_tired, 0, n_rbsf, n_rbsf)
d_tired_rbsf = calculate_cohens_d(mean_rbsf_tired, 0, se_rbsf_tired, 0, n_rbsf, n_rbsf)

t_tired_rb, p_tired_rb = calculate_t(mean_rb_tired, 0, se_rb_tired, 0, n_rb, n_rb)
d_tired_rb = calculate_cohens_d(mean_rb_tired, 0, se_rb_tired, 0, n_rb, n_rb)

# Calculate t, p, and d values for 'jittery' item
t_jittery_placebo, p_jittery_placebo = calculate_t(mean_placebo_jittery, 0, se_placebo_jittery, 0, n_placebo,
                                                   n_placebo)
d_jittery_placebo = calculate_cohens_d(mean_placebo_jittery, 0, se_placebo_jittery, 0, n_placebo, n_placebo)

t_jittery_rbsf, p_jittery_rbsf = calculate_t(mean_rbsf_jittery, 0, se_rbsf_jittery, 0, n_rbsf, n_rbsf)
d_jittery_rbsf = calculate_cohens_d(mean_rbsf_jittery, 0, se_rbsf_jittery, 0, n_rbsf, n_rbsf)

t_jittery_rb, p_jittery_rb = calculate_t(mean_rb_jittery, 0, se_rb_jittery, 0, n_rb, n_rb)
d_jittery_rb = calculate_cohens_d(mean_rb_jittery, 0, se_rb_jittery, 0, n_rb, n_rb)

# Print results
print(f"Alert (Placebo): t={t_alert_placebo}, p={p_alert_placebo}, d={d_alert_placebo}")
print(f"Alert (RBSF): t={t_alert_rbsf}, p={p_alert_rbsf}, d={d_alert_rbsf}")
print(f"Alert (RB): t={t_alert_rb}, p={p_alert_rb}, d={d_alert_rb}")

print(f"Tired (Placebo): t={t_tired_placebo}, p={p_tired_placebo}, d={d_tired_placebo}")
print(f"Tired (RBSF): t={t_tired_rbsf}, p={p_tired_rbsf}, d={d_tired_rbsf}")
print(f"Tired (RB): t={t_tired_rb}, p={p_tired_rb}, d={d_tired_rb}")

print(f"Jittery (Placebo): t={t_jittery_placebo}, p={p_jittery_placebo}, d={d_jittery_placebo}")
print(f"Jittery (RBSF): t={t_jittery_rbsf}, p={p_jittery_rbsf}, d={d_jittery_rbsf}")
print(f"Jittery (RB): t={t_jittery_rb}, p={p_jittery_rb}, d={d_jittery_rb}")
