# EXOCAL: Exoplanet Candidate Assessment and Labeling

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue)](https://www.spaceappschallenge.org/2025/find-a-team/nasa_was_taken1/?tab=project)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A physics-informed, calibrated ML pipeline for reliable exoplanet detection using NASA's open data archives.**

## Overview

EXOCAL reads NASA Exoplanet Archive tables (TOI, KOI, K2) and outputs calibrated probabilities that candidates are real exoplanets, with uncertainty quantification and explanations. Works even when datasets lack false positives [attached_file:1].

**Goal**: Train on NASA open data to analyze new observations and identify exoplanets with reliable confidence scores.

**Pipeline**: Physics-derived features → robust scaling → stacked ensemble (Logistic Regression + Gradient Boosting) → probability calibration → per-target diagnostics (uncertainty + local explanations).

## Technical Approach

### Physics-Informed Features
Derives dimensionless, scale-stable features from transit geometry and radiative scaling [attached_file:1]:
- **Depth fraction**: \(f_d = \frac{\text{depth (ppm)}}{10^6}\)
- **Radius ratio**: \(\frac{R_p}{R_\star} \approx \sqrt{f_d}\)
- **Scaled semi-major axis**: \(\frac{a}{R_\star} \propto \frac{P^{2/3}}{R_\star M_\star^{1/3}}\)
- **Equilibrium temperature**: \(T_{eq} \approx T_{eff}\sqrt{\frac{1}{2}}\sqrt{\frac{R_\star}{a}}\)
- Plus magnitudes, SNR proxies, impact parameters, and quality flags

### Learning Regimes
- **Supervised** (TOI): Standard classification with labeled positives/negatives
- **Positive-Only** (KOI): One-class approach using synthetic background and internal calibration
- **PU Bagging**: Optional ensemble method for datasets with unlabeled samples [attached_file:1]

### Calibrated Ensemble
Stacked classifier using Logistic Regression, Histogram Gradient Boosting, Random Forest, and SVM with Platt or Isotonic calibration for reliable probability outputs [attached_file:1].

## Key Features

- **Calibrated probabilities** with uncertainty quantification
- **Target-level diagnostics**: feature contributions, Monte Carlo uncertainty, PCA neighborhoods
- **Dataset-level validation**: ROC/PR curves, reliability plots, threshold tables
- **Explainable predictions** using local surrogate models and counterfactual analysis
- **Robust preprocessing** with median/IQR standardization handling missing data

## Performance

- **ROC AUC**: >0.94 across survey datasets
- **Calibrated accuracy**: >92% with Expected Calibration Error <15%
- **Processing**: Handles thousands of candidates with per-target uncertainty and explanations [attached_file:1]

## Documentation

For complete technical details, see the attached documentation:
- **Technical specification**: `exo_analysis_wp.pdf` - Mathematical formulation and computational methods
- **Presentation overview**: `exo_analysis_presentation.pdf` - Key concepts and results summary

## NASA Space Apps Challenge 2025

Developed for the **"A World Away: Hunting for Exoplanets with AI"** challenge. 

**Project Page**: [NASA Space Apps 2025 - Team nasa_was_taken1](https://www.spaceappschallenge.org/2025/find-a-team/nasa_was_taken1/?tab=project)

## Repository Structure

Exo-CAL/
├── exocal/ # Core ML pipeline
├── data/ # NASA archive processing
├── notebooks/ # Analysis and validation
├── docs/ # Technical documentation
└── outputs/ # Results and dashboards


---

*"In the vast cosmic ocean, every signal tells a story. EXOCAL helps us listen to the whispers of distant worlds."*
