# Navigation, Guidance and Estimation Algorithms

[![Matlab](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Affiliation](https://img.shields.io/badge/IIT-Bombay-blue.svg)](https://www.iitb.ac.in/)

Implementation and analysis of state estimation and missile guidance laws, focusing on **Kalman Filters** and **Deviated Pursuit**.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `Q1_KalmanFilter.m` | MATLAB script for Discrete-Time Kalman Filter simulation. |
| `Q2_Guidance.m` | MATLAB simulation for Missile-Target engagement (Deviated Pursuit). |
| `Project_Report.pdf` | Detailed technical report with derivations and analysis. |
| `image_4c2101.png` | Geometry and engagement visualization. |

---

## 🚀 Execution Guide

### 1. State Estimation (Problem 1)
Run `Q1_KalmanFilter.m` to simulate a 2nd-order LTI system.
* **Objective:** Estimate states $x_1, x_2$ from noisy measurements $y(k) = Hx(k) + v(k)$.
* **Key Components:** Process noise $Q = BB^T$, Measurement noise $R=1$, and Observability analysis for $H = [2, 2]$.
* **Output:** Comparison of True vs. Estimated states and convergence of error covariance $P(k)$.



### 2. Guidance Simulation (Problem 2)
Run `Q2_Guidance.m` to analyze missile interception dynamics.
* **Guidance Law:** Deviated Pursuit.
* **Scenarios:** The simulation covers 5 cases varying the speed ratio ($\nu$) and deviation angle ($\delta$).
* **Stability Check:** Validates the condition $|\nu \sin \delta| < 1$. 



---

## 📊 Summary of Results

| Scenario | Parameters | Stability | Observations |
| :--- | :--- | :--- | :--- |
| **2, 3, 4** | \nu \sin \delta as in problem_statement | **Stable** | Smooth interception; bounded lateral acceleration. |
| **1, 5** | Boundary/Violated | **Unstable** | Terminal acceleration $a_M$ diverges near intercept. |

> **Conclusion:** Deviated pursuit improves interception geometry over pure pursuit, provided the speed ratio and deviation angle stay within the stable operational region.

---

## 👥 Authors
* **Ayush Bhaskar** (23B0015)
* **Dawar Jyoti Deka** (23B0036)
* **Instructor:** Prof. Arnab Maity, **IIT Bombay**
