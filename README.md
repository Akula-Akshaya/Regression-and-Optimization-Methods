# Regression and Optimization

This repository contains course work on **regression models and optimization techniques**, implemented and analyzed using real datasets. The focus is on understanding how different optimization methods and regression model complexities affect convergence behavior and predictive performance.

---

## Work Included

- Regression analysis on the **Bike Sharing Demand** dataset
- Optimization-based regression for **Earthquake Alert Prediction**
- Comparison of optimization methods under unconstrained and constrained settings
- Empirical evaluation using objective convergence, error metrics, and feature importance

---

## Regression: Bike Sharing Demand

**Objective**  
Predict hourly bike rental demand and study the effect of increasing nonlinearity in regression models.

**Approach**
- Linear Regression (baseline)
- Polynomial Regression (degree 2, 3, 4) without interaction terms
- Polynomial Regression (degree 2) with interaction terms
- Ridge regularization with cross-validation
- Model selection strictly based on **test-set MSE and R²**

**Key Techniques**
- Time-based feature extraction (hour, weekday, month)
- Cyclical encoding using sin/cos transformations
- Standardization and one-hot encoding
- Log-transform of target with smearing correction

**Result**
- Degree-3 polynomial regression (without interactions) achieved the best bias–variance tradeoff and lowest test MSE.

---

## Optimization: Earthquake Alert Prediction

**Objective**  
Study and compare unconstrained and constrained optimization methods on a regression-based alert prediction task.

**Dataset Features**
- Magnitude
- Depth
- CDI (Community Determined Intensity)
- MMI (Modified Mercalli Intensity)

Alert levels were encoded numerically and scaled for regression.

---

### Unconstrained Optimization

- Objective: Regularized Mean Squared Error
- Methods:
  - Gradient Descent
  - Newton’s Method
- Observations:
  - Gradient Descent showed steady but slow convergence
  - Newton’s Method converged in very few iterations due to second-order information

---

### Constrained Optimization

- Objective: Mean Squared Error
- Constraints:
  - ‖w‖² ≤ 1
  - Sum of feature weights = 0
- Method:
  - Quadratic Penalty Method
- Behavior:
  - Early iterations prioritize objective minimization
  - Later iterations strongly enforce constraints as penalty increases

---

## Analysis Highlights

- Newton’s Method converges significantly faster than Gradient Descent
- Penalty Method demonstrates the trade-off between feasibility and objective minimization
- Feature importance remains consistent across methods
- MMI is the most influential feature for earthquake alert prediction

---

## Team

This work was done as part of a course assignment by:

- Akshaya  
- Geethika
