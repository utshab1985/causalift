import pandas as pd
import numpy as np
from causalift import CausalLift

np.random.seed(42)
n = 10000

print("=== TEST 1: Weak confounding ===")
stress_weak = np.random.normal(0, 1, n)
overtime_weak = (0.1 * stress_weak + 
                np.random.normal(0, 1, n)) > 0.5
overtime_weak = overtime_weak.astype(int)
left_weak = (0.4 * overtime_weak + 
             0.6 * stress_weak + 
             np.random.normal(0, 1, n)) > 1
left_weak = left_weak.astype(int)

df_weak = pd.DataFrame({
    'overtime': overtime_weak,
    'stress': stress_weak,
    'left': left_weak
})

model_weak = CausalLift(
    treatment='overtime',
    outcome='left',
    confounders=['stress']
)
model_weak.fit(df_weak).summary()

print("\n=== TEST 2: Strong confounding ===")
stress_strong = np.random.normal(0, 1, n)
overtime_strong = (0.9 * stress_strong + 
                  np.random.normal(0, 0.3, n)) > 0.5
overtime_strong = overtime_strong.astype(int)
left_strong = (0.4 * overtime_strong + 
               0.6 * stress_strong + 
               np.random.normal(0, 1, n)) > 1
left_strong = left_strong.astype(int)

df_strong = pd.DataFrame({
    'overtime': overtime_strong,
    'stress': stress_strong,
    'left': left_strong
})

model_strong = CausalLift(
    treatment='overtime',
    outcome='left',
    confounders=['stress']
)
model_strong.fit(df_strong).summary()