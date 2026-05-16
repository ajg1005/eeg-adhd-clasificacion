# Comparacion estadistica de modelos

## Resumen por modelo (CV cross-subject)

| Familia   | Modelo              |   n_folds |   F1_epoch_mean |   F1_epoch_std |   F1_epoch_ci_low |   F1_epoch_ci_high |   BalancedAccuracy_epoch_mean |   BalancedAccuracy_epoch_std |   BalancedAccuracy_epoch_ci_low |   BalancedAccuracy_epoch_ci_high |
|:----------|:--------------------|----------:|----------------:|---------------:|------------------:|-------------------:|------------------------------:|-----------------------------:|--------------------------------:|---------------------------------:|
| DL        | cnn_1d              |         5 |          0.801  |         0.0732 |            0.7401 |             0.8529 |                        0.8027 |                       0.0737 |                          0.7443 |                           0.8551 |
| DL        | cnn_lstm            |         5 |          0.7924 |         0.0266 |            0.7719 |             0.815  |                        0.7938 |                       0.0231 |                          0.7764 |                           0.8111 |
| ML        | random_forest       |         5 |          0.7872 |         0.0578 |            0.7415 |             0.8329 |                        0.7825 |                       0.0558 |                          0.7381 |                           0.8268 |
| ML        | xgboost             |         5 |          0.7775 |         0.0632 |            0.7255 |             0.8301 |                        0.7735 |                       0.0594 |                          0.7255 |                           0.8203 |
| ML        | rbf_svc             |         5 |          0.752  |         0.0448 |            0.7234 |             0.7926 |                        0.7489 |                       0.0467 |                          0.7217 |                           0.791  |
| ML        | knn                 |         5 |          0.7338 |         0.0589 |            0.6893 |             0.7811 |                        0.7275 |                       0.0611 |                          0.681  |                           0.7772 |
| DL        | eegnet              |         5 |          0.7004 |         0.0948 |            0.6164 |             0.7658 |                        0.7253 |                       0.0815 |                          0.6557 |                           0.7844 |
| ML        | logistic_regression |         5 |          0.6878 |         0.0779 |            0.6277 |             0.751  |                        0.6866 |                       0.0839 |                          0.6245 |                           0.7538 |

CI 95% bootstrap con 10000 resamples (seed=42).

## Tests pareados (ttest_rel sobre folds)

| metric                 | model_a             | model_b       |   n_folds |   mean_diff |   t_stat |   p_value |   p_value_bonferroni | significant_bonferroni_5pct   |
|:-----------------------|:--------------------|:--------------|----------:|------------:|---------:|----------:|---------------------:|:------------------------------|
| F1_epoch               | logistic_regression | rbf_svc       |         5 |     -0.0642 |  -2.845  |    0.0466 |               1      | False                         |
| F1_epoch               | logistic_regression | knn           |         5 |     -0.046  |  -1.0774 |    0.342  |               1      | False                         |
| F1_epoch               | logistic_regression | random_forest |         5 |     -0.0994 |  -3.5098 |    0.0247 |               0.6909 | False                         |
| F1_epoch               | logistic_regression | xgboost       |         5 |     -0.0897 |  -2.5373 |    0.0642 |               1      | False                         |
| F1_epoch               | logistic_regression | eegnet        |         5 |     -0.0126 |  -0.3552 |    0.7404 |               1      | False                         |
| F1_epoch               | logistic_regression | cnn_1d        |         5 |     -0.1132 |  -2.2524 |    0.0874 |               1      | False                         |
| F1_epoch               | logistic_regression | cnn_lstm      |         5 |     -0.1046 |  -2.8809 |    0.045  |               1      | False                         |
| F1_epoch               | rbf_svc             | knn           |         5 |      0.0181 |   0.7748 |    0.4817 |               1      | False                         |
| F1_epoch               | rbf_svc             | random_forest |         5 |     -0.0352 |  -2.1675 |    0.0961 |               1      | False                         |
| F1_epoch               | rbf_svc             | xgboost       |         5 |     -0.0255 |  -0.9948 |    0.3761 |               1      | False                         |
| F1_epoch               | rbf_svc             | eegnet        |         5 |      0.0516 |   1.349  |    0.2486 |               1      | False                         |
| F1_epoch               | rbf_svc             | cnn_1d        |         5 |     -0.049  |  -1.3196 |    0.2574 |               1      | False                         |
| F1_epoch               | rbf_svc             | cnn_lstm      |         5 |     -0.0404 |  -1.7604 |    0.1531 |               1      | False                         |
| F1_epoch               | knn                 | random_forest |         5 |     -0.0534 |  -3.217  |    0.0324 |               0.9064 | False                         |
| F1_epoch               | knn                 | xgboost       |         5 |     -0.0436 |  -2.1262 |    0.1006 |               1      | False                         |
| F1_epoch               | knn                 | eegnet        |         5 |      0.0334 |   0.876  |    0.4305 |               1      | False                         |
| F1_epoch               | knn                 | cnn_1d        |         5 |     -0.0672 |  -2.0778 |    0.1063 |               1      | False                         |
| F1_epoch               | knn                 | cnn_lstm      |         5 |     -0.0586 |  -2.3457 |    0.0789 |               1      | False                         |
| F1_epoch               | random_forest       | xgboost       |         5 |      0.0097 |   0.7812 |    0.4783 |               1      | False                         |
| F1_epoch               | random_forest       | eegnet        |         5 |      0.0868 |   3.3003 |    0.0299 |               0.8379 | False                         |
| F1_epoch               | random_forest       | cnn_1d        |         5 |     -0.0138 |  -0.3692 |    0.7307 |               1      | False                         |
| F1_epoch               | random_forest       | cnn_lstm      |         5 |     -0.0052 |  -0.2072 |    0.846  |               1      | False                         |
| F1_epoch               | xgboost             | eegnet        |         5 |      0.0771 |   2.9279 |    0.0429 |               1      | False                         |
| F1_epoch               | xgboost             | cnn_1d        |         5 |     -0.0235 |  -0.5359 |    0.6205 |               1      | False                         |
| F1_epoch               | xgboost             | cnn_lstm      |         5 |     -0.0149 |  -0.5124 |    0.6354 |               1      | False                         |
| F1_epoch               | eegnet              | cnn_1d        |         5 |     -0.1006 |  -2.2464 |    0.088  |               1      | False                         |
| F1_epoch               | eegnet              | cnn_lstm      |         5 |     -0.092  |  -2.4865 |    0.0677 |               1      | False                         |
| F1_epoch               | cnn_1d              | cnn_lstm      |         5 |      0.0086 |   0.3889 |    0.7171 |               1      | False                         |
| BalancedAccuracy_epoch | logistic_regression | rbf_svc       |         5 |     -0.0623 |  -2.6663 |    0.056  |               1      | False                         |
| BalancedAccuracy_epoch | logistic_regression | knn           |         5 |     -0.0409 |  -0.8977 |    0.4201 |               1      | False                         |
| BalancedAccuracy_epoch | logistic_regression | random_forest |         5 |     -0.0959 |  -3.1666 |    0.034  |               0.9512 | False                         |
| BalancedAccuracy_epoch | logistic_regression | xgboost       |         5 |     -0.0869 |  -2.3691 |    0.0769 |               1      | False                         |
| BalancedAccuracy_epoch | logistic_regression | eegnet        |         5 |     -0.0387 |  -1.0662 |    0.3464 |               1      | False                         |
| BalancedAccuracy_epoch | logistic_regression | cnn_1d        |         5 |     -0.1161 |  -2.1618 |    0.0967 |               1      | False                         |
| BalancedAccuracy_epoch | logistic_regression | cnn_lstm      |         5 |     -0.1072 |  -2.8912 |    0.0445 |               1      | False                         |
| BalancedAccuracy_epoch | rbf_svc             | knn           |         5 |      0.0214 |   0.8195 |    0.4585 |               1      | False                         |
| BalancedAccuracy_epoch | rbf_svc             | random_forest |         5 |     -0.0336 |  -1.9866 |    0.1179 |               1      | False                         |
| BalancedAccuracy_epoch | rbf_svc             | xgboost       |         5 |     -0.0246 |  -0.9343 |    0.4031 |               1      | False                         |
| BalancedAccuracy_epoch | rbf_svc             | eegnet        |         5 |      0.0235 |   0.6989 |    0.5231 |               1      | False                         |
| BalancedAccuracy_epoch | rbf_svc             | cnn_1d        |         5 |     -0.0538 |  -1.4183 |    0.2291 |               1      | False                         |
| BalancedAccuracy_epoch | rbf_svc             | cnn_lstm      |         5 |     -0.0449 |  -2.2248 |    0.0901 |               1      | False                         |
| BalancedAccuracy_epoch | knn                 | random_forest |         5 |     -0.055  |  -3.3233 |    0.0293 |               0.8201 | False                         |
| BalancedAccuracy_epoch | knn                 | xgboost       |         5 |     -0.046  |  -2.2275 |    0.0899 |               1      | False                         |
| BalancedAccuracy_epoch | knn                 | eegnet        |         5 |      0.0022 |   0.0715 |    0.9464 |               1      | False                         |
| BalancedAccuracy_epoch | knn                 | cnn_1d        |         5 |     -0.0752 |  -2.2566 |    0.087  |               1      | False                         |
| BalancedAccuracy_epoch | knn                 | cnn_lstm      |         5 |     -0.0663 |  -2.5457 |    0.0636 |               1      | False                         |
| BalancedAccuracy_epoch | random_forest       | xgboost       |         5 |      0.009  |   0.6961 |    0.5247 |               1      | False                         |
| BalancedAccuracy_epoch | random_forest       | eegnet        |         5 |      0.0571 |   2.9092 |    0.0437 |               1      | False                         |
| BalancedAccuracy_epoch | random_forest       | cnn_1d        |         5 |     -0.0202 |  -0.5402 |    0.6177 |               1      | False                         |
| BalancedAccuracy_epoch | random_forest       | cnn_lstm      |         5 |     -0.0113 |  -0.471  |    0.6622 |               1      | False                         |
| BalancedAccuracy_epoch | xgboost             | eegnet        |         5 |      0.0481 |   2.6986 |    0.0542 |               1      | False                         |
| BalancedAccuracy_epoch | xgboost             | cnn_1d        |         5 |     -0.0292 |  -0.6601 |    0.5453 |               1      | False                         |
| BalancedAccuracy_epoch | xgboost             | cnn_lstm      |         5 |     -0.0203 |  -0.6958 |    0.5249 |               1      | False                         |
| BalancedAccuracy_epoch | eegnet              | cnn_1d        |         5 |     -0.0773 |  -1.7599 |    0.1532 |               1      | False                         |
| BalancedAccuracy_epoch | eegnet              | cnn_lstm      |         5 |     -0.0684 |  -1.9645 |    0.1209 |               1      | False                         |
| BalancedAccuracy_epoch | cnn_1d              | cnn_lstm      |         5 |      0.0089 |   0.3791 |    0.7239 |               1      | False                         |

p_value_bonferroni = p crudo * numero de pares; significant_bonferroni_5pct = True si p_bonf < 0.05.