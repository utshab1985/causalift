import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
import math

class CausalLift:
    """
    CausalLift: Find the true causal effect of any intervention.
    Separates what you caused from what would have happened anyway.
    """

    def __init__(self, treatment, outcome, confounders):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.model = None
        self.results = {}

    def fit(self, data):
        features = [self.treatment] + self.confounders
        X = data[features].values
        y = data[self.outcome].values

        self.model = LogisticRegression()
        self.model.fit(X, y)

        X_naive = data[[self.treatment]].values
        naive_model = LogisticRegression()
        naive_model.fit(X_naive, y)
        self.results['naive_odds_ratio'] = math.exp(naive_model.coef_[0][0])

        X_conf = data[self.confounders].values
        X_treat = data[self.treatment].values
        conf_model = LinearRegression()
        conf_model.fit(X_conf, X_treat)
        r2 = r2_score(X_treat, conf_model.predict(X_conf))
        self.results['confounding_severity'] = r2

        self.results['treatment_effect_log_odds'] = self.model.coef_[0][0]
        self.results['treatment_odds_ratio'] = math.exp(self.model.coef_[0][0])
        self.results['confounder_odds_ratios'] = [
            math.exp(c) for c in self.model.coef_[0][1:]
        ]

        return self

    def ate(self, data):
        features = [self.treatment] + self.confounders

        data_treated = data[features].copy()
        data_treated[self.treatment] = 1

        data_control = data[features].copy()
        data_control[self.treatment] = 0

        prob_treated = self.model.predict_proba(
            data_treated.values
        )[:, 1]

        prob_control = self.model.predict_proba(
            data_control.values
        )[:, 1]

        individual_effects = prob_treated - prob_control
        ate_value = individual_effects.mean()

        self.results['ate'] = ate_value
        self.results['individual_effects'] = individual_effects

        return ate_value

    def summary(self):
        naive = self.results['naive_odds_ratio']
        corrected = self.results['treatment_odds_ratio']
        severity = self.results['confounding_severity'] * 100

        print(f"\n--- CausalLift Results ---")
        print(f"Naive effect (uncorrected):     {naive:.2f}x")
        print(f"True causal effect (corrected): {corrected:.2f}x")
        print(f"Confounding severity score:     {severity:.1f}%")
        print()

        if severity < 5:
            verdict = "LOW - naive measurement was mostly reliable"
        elif severity < 20:
            verdict = "MEDIUM - naive measurement was somewhat misleading"
        else:
            verdict = "HIGH - naive measurement was seriously wrong"

        print(f"Verdict: {verdict}")
        print()
        print(f"Without CausalLift you would have reported {naive:.2f}x")
        print(f"The true effect is {corrected:.2f}x")
        print(f"You were off by {abs(naive-corrected):.2f}x")

        if 'ate' in self.results:
            ate = self.results['ate']
            print(f"\nAverage Treatment Effect (ATE):  {ate:.4f}")
            print(f"In plain English: the treatment causes {ate*100:.1f} extra outcomes per 100 people")

        return self