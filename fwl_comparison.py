import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv('hr_data.csv')

X_stress = df[['stress_score']].values
y_overtime = df['overtime'].values
y_left = df['left_company'].values

# ---- METHOD 1: Linear in Step 1 ----
step1_linear = LinearRegression()
step1_linear.fit(X_stress, y_overtime)
predicted_linear = step1_linear.predict(X_stress)
residual_linear = y_overtime - predicted_linear

step3_linear = LinearRegression()
step3_linear.fit(residual_linear.reshape(-1, 1), y_left)
m1_linear = step3_linear.coef_[0]

# ---- METHOD 2: Logistic in Step 1 ----
step1_logistic = LogisticRegression()
step1_logistic.fit(X_stress, y_overtime)
predicted_logistic = step1_logistic.predict_proba(X_stress)[:, 1]
residual_logistic = y_overtime - predicted_logistic

step3_logistic = LinearRegression()
step3_logistic.fit(residual_logistic.reshape(-1, 1), y_left)
m1_logistic = step3_logistic.coef_[0]

# ---- DIRECT REGRESSION (ground truth) ----
X_both = df[['overtime', 'stress_score']].values
direct = LinearRegression()
direct.fit(X_both, y_left)
m1_direct = direct.coef_[0]

print("--- FWL Comparison ---")
print(f"Direct regression m1:        {m1_direct:.6f}")
print(f"FWL with linear Step 1 m1:   {m1_linear:.6f}")
print(f"FWL with logistic Step 1 m1: {m1_logistic:.6f}")
print()
print(f"Linear error vs direct:   {abs(m1_linear - m1_direct):.6f}")
print(f"Logistic error vs direct: {abs(m1_logistic - m1_direct):.6f}")
print()
print("Which is closer to ground truth?")
if abs(m1_logistic - m1_direct) < abs(m1_linear - m1_direct):
    print("Logistic Step 1 wins")
else:
    print("Linear Step 1 wins")

print("\n--- What happens when stress STRONGLY predicts overtime? ---")

np.random.seed(42)
n = 10000
stress_strong = np.random.normal(0, 1, n)

# Now stress strongly drives overtime
overtime_strong = (0.8 * stress_strong + 
                  np.random.normal(0, 0.5, n)) > 0
overtime_strong = overtime_strong.astype(int)

left_strong = (0.4 * overtime_strong + 
               0.6 * stress_strong + 
               np.random.normal(0, 1, n)) > 1
left_strong = left_strong.astype(int)

df_strong = pd.DataFrame({
    'overtime': overtime_strong,
    'stress_score': stress_strong,
    'left_company': left_strong
})

X_stress_s = df_strong[['stress_score']].values
y_overtime_s = df_strong['overtime'].values
y_left_s = df_strong['left_company'].values

# Linear Step 1
s1_lin = LinearRegression()
s1_lin.fit(X_stress_s, y_overtime_s)
res_lin = y_overtime_s - s1_lin.predict(X_stress_s)
s3_lin = LinearRegression()
s3_lin.fit(res_lin.reshape(-1, 1), y_left_s)

# Logistic Step 1
s1_log = LogisticRegression()
s1_log.fit(X_stress_s, y_overtime_s)
res_log = y_overtime_s - s1_log.predict_proba(X_stress_s)[:, 1]
s3_log = LinearRegression()
s3_log.fit(res_log.reshape(-1, 1), y_left_s)

# Direct
X_both_s = df_strong[['overtime', 'stress_score']].values
dir_s = LinearRegression()
dir_s.fit(X_both_s, y_left_s)

print(f"Direct regression m1:        {dir_s.coef_[0]:.6f}")
print(f"FWL linear Step 1 m1:        {s3_lin.coef_[0]:.6f}")
print(f"FWL logistic Step 1 m1:      {s3_log.coef_[0]:.6f}")
print(f"Linear error:  {abs(s3_lin.coef_[0] - dir_s.coef_[0]):.6f}")
print(f"Logistic error: {abs(s3_log.coef_[0] - dir_s.coef_[0]):.6f}")

if abs(s3_log.coef_[0] - dir_s.coef_[0]) < abs(s3_lin.coef_[0] - dir_s.coef_[0]):
    print("Logistic Step 1 wins under strong correlation")
else:
    print("Linear Step 1 still wins under strong correlation")