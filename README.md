# Proactive Handover Type Prediction and Parameter Optimization Based on Machine Learning
# Description
This paper proposes a scheme that proactively predicts HO types and dynamically adjusts HCPs based on RL and supervised learning, and performs a performance analysis on this scheme.
This code provides a simulation scenario based on A3 event handover. It also gives a parameter TTT and HOM optimization scheme. Some code is provided in the mian file for the reader to study. Since there is too much data, only some of the data files are shown here for the reader's discussion. If you need detailed data files please contact email hqf6800@gmail.com.

# Dependencies
- python 3.9.13
- scikit-learn 1.4.2
- tensorflow 2.16.1
- tensorflow-gpu 2.6.0
- xgboost 2.0.3
# Usage
This simulator can consider also the movement of UEs:
-   UEs have fixed speed and there are two possible movements implemented at this time: 
    -   random movement
    -   line movement given a direction, with bumping on the borders of the map
- HO_initialize: This includes all functions where the user triggers an event switch based on A3.

- globalval: This includes all the parameter settings of the reference user in the simulation.

- THA: User handover in fixed  parameters TTT and HOM.

- SLEHA: User handover with predictions and adjust TTT and HOM to (0, 5), (10, 9).
 
- QHA: Optimizing TTT and HOM with Q-learing

- DQN+prediction: The proposed scheme in this paper, using DQN combined with supervised learning for HCPs optimization.

- handover_optimal without ML: No use of machine learning to tune HCPs.

- Classifiers_Comparison : Comparison of prediction performance of different classifiers.

- XGB: Predictive accuracy analysis of XGB classifiers alone.

If you use this code or any herein modified part of it in any publication, please cite the paper:


