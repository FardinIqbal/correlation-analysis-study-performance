"""
Cleaned-up version of the code extracted from the Jupyter Notebook.
Refactored for readability and clarity, with unnecessary repetition removed.
Functions have been added where appropriate.
"""

import numpy as np
from scipy.stats import rankdata, spearmanr

# Data: Hours of Study (X) and Exam Scores (Y)
X = np.array([5, 8, 7, 9, 11, 4.5, 10, 3, 12, 6, 7, 10, 2, 13, 8, 14, 5.5, 3.5, 12.5, 11.5])
Y = np.array([62, 74, 69, 76, 85, 60, 80, 55, 90, 65, 70, 82, 50, 92, 78, 95, 63, 58, 88, 86])

# Function to compute Pearson correlation and related statistics
def compute_statistics(X, Y, dataset_name="Dataset"):
    mean_X, mean_Y = np.mean(X), np.mean(Y)
    std_X, std_Y = np.std(X, ddof=1), np.std(Y, ddof=1)  # Sample standard deviations
    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    pearson_r = np.corrcoef(X, Y)[0, 1]
    correlation_strength = "strong" if abs(pearson_r) > 0.7 else "moderate" if abs(pearson_r) > 0.4 else "weak"
    correlation_direction = "positive" if pearson_r > 0 else "negative"

    print("=" * 60)
    print(f"{dataset_name:^60}")
    print("=" * 60)
    print(f"Mean of X: {mean_X:.2f}, Mean of Y: {mean_Y:.2f}")
    print(f"Standard Deviation of X: {std_X:.2f}, Standard Deviation of Y: {std_Y:.2f}")
    print(f"Covariance: {cov_XY:.2f}")
    print(f"Pearson Correlation Coefficient (r): {pearson_r:.3f}")
    print(f"Correlation: {correlation_strength.capitalize()} & {correlation_direction.capitalize()}")
    print("=" * 60)

# Compute and print Pearson correlation statistics
compute_statistics(X, Y, dataset_name="Study Hours vs Exam Scores")

# Compute Spearman's Rank Correlation
ranks_X = rankdata(X)
ranks_Y = rankdata(Y)
d_squared = (ranks_X - ranks_Y) ** 2
sum_d_squared = np.sum(d_squared)
spearman_rho, _ = spearmanr(X, Y)
spearman_strength = "strong" if abs(spearman_rho) > 0.7 else "moderate" if abs(spearman_rho) > 0.4 else "weak"
spearman_direction = "positive" if spearman_rho > 0 else "negative"

print("=" * 70)
print(f"{'Spearman Rank Correlation Analysis':^70}")
print("=" * 70)
print(f"Ranks for X: {ranks_X}")
print(f"Ranks for Y: {ranks_Y}")
print(f"Sum of Squared Differences (Σ d²): {sum_d_squared}")
print(f"Spearman Rank Correlation Coefficient (ρ): {spearman_rho:.3f}")
print(f"Correlation: {spearman_strength.capitalize()} & {spearman_direction.capitalize()}")
print("=" * 70)

# Define distributions for mean and standard deviation comparisons
distributions = {
    "(a) i": np.array([3, 5, 5, 5, 8, 11, 11, 11, 13]),
    "(a) ii": np.array([3, 5, 5, 5, 8, 11, 11, 11, 20]),
    "(b) i": np.array([-20, 0, 0, 0, 15, 25, 30, 30]),
    "(b) ii": np.array([-40, 0, 0, 0, 15, 25, 30, 30]),
    "(c) i": np.array([0, 2, 4, 6, 8, 10]),
    "(c) ii": np.array([20, 22, 24, 26, 28, 30]),
    "(d) i": np.array([100, 200, 300, 400, 500]),
    "(d) ii": np.array([0, 50, 300, 550, 600])
}

# Compute mean and standard deviation for each distribution
results = {label: {"Mean": np.mean(data), "Std Dev": np.std(data, ddof=1)} for label, data in distributions.items()}

# Compare means and standard deviations for each pair
pairs = [("(a) i", "(a) ii"), ("(b) i", "(b) ii"), ("(c) i", "(c) ii"), ("(d) i", "(d) ii")]

print("=" * 50)
print(f"{'Comparison of Means and Standard Deviations':^50}")
print("=" * 50)
for pair in pairs:
    mean_winner = pair[0] if results[pair[0]]["Mean"] > results[pair[1]]["Mean"] else pair[1]
    std_dev_winner = pair[0] if results[pair[0]]["Std Dev"] > results[pair[1]]["Std Dev"] else pair[1]
    print(f"Pair: {pair[0]} vs {pair[1]}")
    print(f"  - Greater Mean: {mean_winner}")
    print(f"  - Greater Standard Deviation: {std_dev_winner}")
    print("-" * 50)

# Probability computations for independent events
P_A, P_B = 0.3, 0.7
P_A_and_B = P_A * P_B
P_A_or_B = P_A + P_B - P_A_and_B
P_A_given_B = P_A_and_B / P_B

print(f"P(A and B) = {P_A_and_B:.4f}")
print(f"P(A or B) = {P_A_or_B:.4f}")
print(f"P(A | B) = {P_A_given_B:.4f}")

# Disease Screening Test Data
total_population = 2000
disease_present = 100
disease_absent = 1900
test_positive = 200
test_negative = 1800
true_positives = 95
false_positives = 105
true_negatives = 1795
false_negatives = 5

# Calculate probabilities for the screening test
P_D_plus = disease_present / total_population
P_T_plus = test_positive / total_population
P_D_plus_given_T_plus = true_positives / test_positive
P_T_plus_given_D_plus = true_positives / disease_present
P_T_minus_given_D_minus = true_negatives / disease_absent

print("=" * 60)
print(f"{'Disease Screening Test Analysis':^60}")
print("=" * 60)
print(f"P(D+) - Prior probability of disease: {P_D_plus:.4f} (5%)")
print(f"P(T+) - Probability of testing positive: {P_T_plus:.4f} (10%)")
print(f"P(D+ | T+) - Probability of having disease given positive test: {P_D_plus_given_T_plus:.4f} (47.5%)")
print(f"P(T+ | D+) - Sensitivity (True Positive Rate): {P_T_plus_given_D_plus:.4f} (95%)")
print(f"P(T- | D-) - Specificity (True Negative Rate): {P_T_minus_given_D_minus:.4f} (94.47%)")
print("=" * 60)
