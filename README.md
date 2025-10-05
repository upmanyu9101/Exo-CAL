What it is.
A physics-informed, calibrated ML pipeline that reads NASA Exoplanet Archive tables (TOI, KOI, K2) and outputs, for every target, a probability of being a real exoplanet, with uncertainty, explanations, and minimal dashboards. Works even when a dataset lacks false positives.

TL;DR 
Goal: Train on NASA open data; analyze new rows to identify exoplanets.
How: Physics-derived features → robust scaling → stacked ensemble (Logistic Regression + Gradient Boosting) → probability calibration → per-target diagnostics (uncertainty + local explanations).
Why it matters: Combines astrophysical priors with calibrated AI so scores are reliable, explainable, and actionable for follow-up.

