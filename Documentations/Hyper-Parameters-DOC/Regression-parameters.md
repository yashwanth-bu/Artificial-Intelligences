# Hyperparameter Grids for Regression Models

## 1. **Linear Regression**

Linear regression typically doesn't require tuning, but you can experiment with its regularized variants:

### ✅ **Ridge Regression**

```python
{
    'ridge__alpha': [0.01, 0.1, 1.0, 10, 100]
}
```

### ✅ **Lasso Regression**

```python
{
    'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10]
}
```

### ✅ **ElasticNet**

```python
{
    'elasticnet__alpha': [0.001, 0.01, 0.1, 1.0, 10],
    'elasticnet__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
}
```

---

## 2. **Decision Tree Regressor**

```python
{
    'tree__max_depth': [None, 5, 10, 20, 50],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 2, 4],
    'tree__criterion': ['squared_error', 'absolute_error']
}
```

---

## 3. **Random Forest Regressor**

```python
{
    'rf__n_estimators': [100, 200, 500],
    'rf__max_depth': [None, 10, 20, 50],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['auto', 'sqrt', 'log2']
}
```

---

## 4. **Gradient Boosting Regressor**

```python
{
    'gb__n_estimators': [100, 200, 300],
    'gb__learning_rate': [0.01, 0.05, 0.1],
    'gb__max_depth': [3, 5, 10],
    'gb__min_samples_split': [2, 5, 10],
    'gb__min_samples_leaf': [1, 2, 4],
    'gb__subsample': [0.5, 0.8, 1.0]
}
```

---

## 5. **XGBoost Regressor** (`xgboost.XGBRegressor`)

```python
{
    'xgb__n_estimators': [100, 200],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__max_depth': [3, 6, 10],
    'xgb__subsample': [0.5, 0.8, 1.0],
    'xgb__colsample_bytree': [0.5, 0.8, 1.0],
    'xgb__reg_alpha': [0, 0.1, 1.0],
    'xgb__reg_lambda': [1.0, 1.5, 2.0]
}
```

---

## 6. **Support Vector Regressor (SVR)**

```python
{
    'svr__kernel': ['linear', 'rbf', 'poly'],
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.2],
    'svr__gamma': ['scale', 'auto']
}
```

### ✅ **Bayesian Ridge**

```python
{
    'bayes__alpha_1': [1e-7, 1e-6, 1e-5],
    'bayes__alpha_2': [1e-7, 1e-6, 1e-5],
    'bayes__lambda_1': [1e-7, 1e-6, 1e-5],
    'bayes__lambda_2': [1e-7, 1e-6, 1e-5]
}
```

### ✅ **Huber Regressor**

```python
{
    'huber__epsilon': [1.1, 1.35, 1.5, 1.75, 2.0],
    'huber__alpha': [0.0001, 0.001, 0.01]
}
```

### ✅ **Passive Aggressive Regressor**

```python
{
    'pa__C': [0.01, 0.1, 1.0, 10],
    'pa__epsilon': [0.01, 0.1, 0.5],
    'pa__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
}
```

---

## 7. **Tree-Based Models**

### ✅ **ExtraTrees Regressor**

```python
{
    'et__n_estimators': [100, 200, 500],
    'et__max_depth': [None, 10, 20, 30],
    'et__min_samples_split': [2, 5, 10],
    'et__min_samples_leaf': [1, 2, 4],
    'et__max_features': ['auto', 'sqrt', 'log2']
}
```

### ✅ **HistGradientBoosting Regressor**

```python
{
    'hgb__learning_rate': [0.01, 0.1, 0.2],
    'hgb__max_iter': [100, 200, 300],
    'hgb__max_depth': [None, 10, 20],
    'hgb__min_samples_leaf': [20, 50, 100],
    'hgb__l2_regularization': [0.0, 0.1, 1.0]
}
```

---

## 8. **Ensemble Models**

### ✅ **AdaBoost Regressor**

```python
{
    'ada__n_estimators': [50, 100, 200],
    'ada__learning_rate': [0.01, 0.1, 1.0],
    'ada__loss': ['linear', 'square', 'exponential']
}
```

### ✅ **Bagging Regressor**

```python
{
    'bag__n_estimators': [10, 50, 100],
    'bag__max_samples': [0.5, 0.7, 1.0],
    'bag__max_features': [0.5, 0.7, 1.0]
}
```

---

## 9. **k-Nearest Neighbors (KNN)**

```python
{
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__p': [1, 2]  # 1=Manhattan, 2=Euclidean
}
```

---

## 10. **MLP Regressor (Neural Network)**

```python
{
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam', 'lbfgs'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate': ['constant', 'adaptive']
}
```

---

## 11. **K-Nearest Neighbors Regressor (KNR)**

```python
{
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__p': [1, 2]  # 1=Manhattan, 2=Euclidean
}
```

---

## 12. **Orthogonal Matching Pursuit (OMP)**

```python
{
    'omp__n_nonzero_coefs': [1, 2, 5, 10, 50],
    'omp__tol': [1e-5, 1e-4, 1e-3],
    'omp__fit_intercept': [True, False]
}
```

---

## 13. **RANSAC Regressor**

```python
{
    'ransac__max_trials': [100, 200, 500],
    'ransac__min_samples': [None, 0.5, 0.75],
    'ransac__residual_threshold': [1.0, 2.0, 5.0],
    'ransac__loss': ['absolute_loss', 'squared_loss']
}
```

---

## 14. **TheilSen Regressor**

```python
{
    'theil_sen__max_iter': [200, 500, 1000],
    'theil_sen__tol': [1e-4, 1e-3, 1e-2],
    'theil_sen__fit_intercept': [True, False]
}
```

---

## 15. **Poisson Regressor**

```python
{
    'poisson__alpha': [0.1, 0.5, 1.0, 2.0],
    'poisson__fit_intercept': [True, False]
}
```

---

## 16. **Quantile Regressor**

```python
{
    'quantile__alpha': [0.1, 0.5, 0.9],
    'quantile__fit_intercept': [True, False]
}
```

---

## 17. **Logistic Regression (for classification tasks with probability regression)**

```python
{
    'logreg__C': [0.01, 0.1, 1.0, 10],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__solver': ['liblinear', 'saga'],
    'logreg__max_iter': [100, 200, 500]
}
```

---

## 18. **ElasticNetCV**

```python
{
    'elasticnetcv__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0],
    'elasticnetcv__alphas': [[0.01, 0.1, 1.0, 10], [0.001, 0.01, 0.1, 1.0]],
    'elasticnetcv__cv': [5, 10]
}
```

---

## 19. **Tweedie Regressor**

```python
{
    'tweedie__power': [1.5, 2.0, 2.5, 3.0],
    'tweedie__alpha': [0.01, 0.1, 1.0],
    'tweedie__link': ['log', 'identity']
}
```

---

## 20. **Bayesian Ridge Regression**

```python
{
    'bayes__alpha_1': [1e-7, 1e-6, 1e-5],
    'bayes__alpha_2': [1e-7, 1e-6, 1e-5],
    'bayes__lambda_1': [1e-7, 1e-6, 1e-5],
    'bayes__lambda_2': [1e-7, 1e-6, 1e-5],
    'bayes__fit_intercept': [True, False]
}
```

---

## 21. **Dummy Regressor (for baseline)**

```python
{
    'dummy__strategy': ['mean', 'median', 'quantile'],
    'dummy__quantile': [0.25, 0.5, 0.75]
}
```

---

## 22. **SGD Regressor**

```python
{
    'sgd__loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'sgd__alpha': [0.0001, 0.001, 0.01],
    'sgd__penalty': ['l2', 'l1', 'elasticnet'],
    'sgd__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'sgd__max_iter': [1000, 2000, 5000]
}
```

---

## 23. **LinearSVR**

```python
{
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.2],
    'svr__kernel': ['linear'],
    'svr__max_iter': [1000, 2000, 5000]
}
```

---

## 24. **RidgeCV (Ridge Regression with built-in cross-validation)**

```python
{
    'ridgecv__alphas': [0.1, 1.0, 10, 100, 1000],
    'ridgecv__store_cv_values': [True, False]
}
```

---

## 25. **Kernel Ridge Regression**

```python
{
    'kernelridge__alpha': [0.1, 1.0, 10],
    'kernelridge__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'kernelridge__degree': [3, 5, 7],
    'kernelridge__gamma': ['scale', 'auto']
}
```

---

## 26. **Gaussian Process Regressor**

```python
{
    'gp__alpha': [1e-2, 1e-1, 1, 10],
    'gp__kernel': ['1.0 * RBF(length_scale=1.0)', '1.0 * RBF(length_scale=10.0)'],
    'gp__n_restarts_optimizer': [0, 10, 20],
    'gp__normalize_y': [True, False]
}
```

---

## 27. **MLP Regressor (Multi-Layer Perceptron)**

```python
{
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (150,)],
    'mlp__activation': ['relu', 'tanh', 'logistic'],
    'mlp__solver': ['adam', 'lbfgs'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__learning_rate_init': [0.001, 0.01],
    'mlp__max_iter': [100, 200, 500]
}
```

---

## 28. **Huber Regressor**

```python
{
    'huber__epsilon': [1.0, 1.1, 1.5, 2.0],
    'huber__alpha': [0.0001, 0.001, 0.01],
    'huber__max_iter': [100, 200, 300],
}
```

---

## 29. **AdaBoost Regressor**

```python
{
    'ada__n_estimators': [50, 100, 200, 500],
    'ada__learning_rate': [0.01, 0.1, 0.5, 1.0],
    'ada__loss': ['linear', 'square', 'exponential']
}
```

---

## 30. **Bagging Regressor**

```python
{
    'bag__n_estimators': [10, 50, 100, 200],
    'bag__max_samples': [0.5, 0.7, 1.0],
    'bag__max_features': [0.5, 0.7, 1.0],
    'bag__bootstrap': [True, False]
}
```

---

## 31. **Stacking Regressor**

```python
{
    'stacking__final_estimator': [LinearRegression(), Ridge(), Lasso()],
    'stacking__cv': [5, 10],
    'stacking__passthrough': [True, False]
}
```

---

## 32. **QuantileRegressor**

```python
{
    'quantile__alpha': [0.1, 0.25, 0.5, 0.75],
    'quantile__fit_intercept': [True, False],
}
```

---

## 33. **ElasticNetCV (ElasticNet with built-in cross-validation)**

```python
{
    'elasticnetcv__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0],
    'elasticnetcv__alphas': [[0.01, 0.1, 1.0, 10], [0.001, 0.01, 0.1, 1.0]],
    'elasticnetcv__cv': [5, 10]
}
```

---

## 34. **Poisson Regression**

```python
{
    'poisson__alpha': [0.1, 1.0, 10],
    'poisson__fit_intercept': [True, False]
}
```

---

## 35. **Tweedie Regressor**

```python
{
    'tweedie__power': [1.5, 2.0, 2.5, 3.0],
    'tweedie__alpha': [0.01, 0.1, 1.0],
    'tweedie__link': ['log', 'identity']
}
```

---

## 36. **Logistic Regression (for classification with probability regression)**

```python
{
    'logreg__C': [0.01, 0.1, 1.0, 10],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__solver': ['liblinear', 'saga'],
    'logreg__max_iter': [100, 200, 500]
}
```

---

## 37. **Ridge Regression (with regularization)**

```python
{
    'ridge__alpha': [0.1, 1.0, 10, 100, 1000],
    'ridge__fit_intercept': [True, False],
    'ridge__normalize': [True, False]
}
```

---

## 38. **Bayesian Ridge Regression**

```python
{
    'bayes__alpha_1': [1e-7, 1e-6, 1e-5],
    'bayes__alpha_2': [1e-7, 1e-6, 1e-5],
    'bayes__lambda_1': [1e-7, 1e-6, 1e-5],
    'bayes__lambda_2': [1e-7, 1e-6, 1e-5],
    'bayes__fit_intercept': [True, False]
}
```

---

## 39. **Dummy Regressor (Baseline model)**

```python
{
    'dummy__strategy': ['mean', 'median', 'quantile'],
    'dummy__quantile': [0.25, 0.5, 0.75]
}
```

---

## 40. **GaussianNB (Naive Bayes for Regression)**

```python
{
    'gnb__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
}
```

---

## 41. **Neural Network Regressor (Custom Model)**

```python
{
    'nn__hidden_layer_sizes': [(50,), (100,), (100, 50), (150,)],
    'nn__activation': ['relu', 'tanh'],
    'nn__solver': ['adam', 'lbfgs'],
    'nn__alpha': [0.0001, 0.001, 0.01],
    'nn__learning_rate': ['constant', 'adaptive'],
    'nn__max_iter': [1000, 2000]
}
```

---

## 42. **Simple Linear Regression**

```python
{
    'linear__fit_intercept': [True, False],
    'linear__normalize': [True, False]
}
```

---

## 43. **DecisionTree Regressor (Improved)**

```python
{
    'tree__max_depth': [None, 5, 10, 20, 50],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 2, 4],
    'tree__max_features': ['auto', 'sqrt', 'log2'],
    'tree__criterion': ['squared_error', 'absolute_error']
}
```

---

## 44. **RandomForest Regressor (Enhanced)**

```python
{
    'rf__n_estimators': [100, 200, 500, 1000],
    'rf__max_depth': [None, 10, 20, 50],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['auto', 'sqrt', 'log2'],
    'rf__bootstrap': [True, False]
}
```

---

## 45. **LightGBM Regressor**

```python
{
    'lgbm__num_leaves': [31, 50, 100],
    'lgbm__max_depth': [-1, 5, 10],
    'lgbm__learning_rate': [0.01, 0.05, 0.1],
    'lgbm__n_estimators': [100, 200, 500],
    'lgbm__subsample': [0.6, 0.8, 1.0],
    'lgbm__colsample_bytree': [0.6, 0.8, 1.0]
}
```

---

## 46. **CatBoost Regressor**

```python
{
    'catboost__iterations': [100, 500, 1000],
    'catboost__learning_rate': [0.01, 0.05, 0.1],
    'catboost__depth': [3, 5, 7],
    'catboost__l2_leaf_reg': [3, 5, 7],
    'catboost__subsample': [0.7, 0.8, 0.9],
    'catboost__colsample_bylevel': [0.5, 0.7, 1.0]
}
```

---

## 47. **XGBoost Regressor (XGB)**

```python
{
    'xgb__n_estimators': [100, 200, 500],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__max_depth': [3, 6, 10],
    'xgb__subsample': [0.5, 0.8, 1.0],
    'xgb__colsample_bytree': [0.5, 0.8, 1.0],
    'xgb__reg_alpha': [0.0, 0.1, 1.0],
    'xgb__reg_lambda': [1.0, 1.5, 2.0]
}
```

---

## 48. **Neural Network Regressor (MLP) with Early Stopping**

```python
{
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (150,)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam', 'lbfgs'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__early_stopping': [True, False],
    'mlp__validation_fraction': [0.1, 0.2, 0.3]
}
```

---

## 49. **HGB (HistGradientBoosting Regressor)**

```python
{
    'hgb__learning_rate': [0.01, 0.05, 0.1],
    'hgb__max_iter': [100, 200, 500],
    'hgb__max_depth': [None, 5, 10],
    'hgb__min_samples_leaf': [10, 20, 50],
    'hgb__l2_regularization': [0.0, 0.1, 1.0]
}
```

---

## 50. **ExtraTrees Regressor**

```python
{
    'et__n_estimators': [100, 200, 500],
    'et__max_depth': [None, 5, 10, 30],
    'et__min_samples_split': [2, 5, 10],
    'et__min_samples_leaf': [1, 2, 4],
    'et__max_features': ['auto', 'sqrt', 'log2'],
    'et__bootstrap': [True, False]
}
```

---

## 51. **K-Nearest Neighbors Regressor (KNN) with Grid Search**

```python
{
    'knn__n_neighbors': [3, 5, 7, 10],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__p': [1, 2],  # 1=Manhattan, 2=Euclidean
    'knn__leaf_size': [30, 50, 100]
}
```

---

## 52. **Ridge Regression (with Cross-validation)**

```python
{
    'ridgecv__alphas': [0.1, 1.0, 10, 100, 1000],
    'ridgecv__store_cv_values': [True, False]
}
```

---

## 53. **SVR (Support Vector Regressor) with Radial Basis Function (RBF) Kernel**

```python
{
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.2],
    'svr__gamma': ['scale', 'auto'],
    'svr__kernel': ['rbf']
}
```

---

## 54. **Gaussian Process Regressor**

```python
{
    'gp__alpha': [1e-2, 1e-1, 1, 10],
    'gp__kernel': ['1.0 * RBF(length_scale=1.0)', '1.0 * RBF(length_scale=10.0)'],
    'gp__n_restarts_optimizer': [0, 10, 20],
    'gp__normalize_y': [True, False],
    'gp__optimizer': ['fmin_l_bfgs_b', 'fmin_tnc']
}
```

---

## 55. **ElasticNet Regressor (with L1/L2 Regularization)**

```python
{
    'elasticnet__alpha': [0.1, 0.5, 1.0],
    'elasticnet__l1_ratio': [0.1, 0.5, 0.7, 1.0],
    'elasticnet__fit_intercept': [True, False],
    'elasticnet__normalize': [True, False]
}
```

---

## 56. **Hinge Loss Regressor (SVM-based)**

```python
{
    'hinge__C': [0.1, 1, 10],
    'hinge__epsilon': [0.01, 0.1, 0.2],
    'hinge__gamma': ['scale', 'auto'],
    'hinge__kernel': ['linear', 'rbf']
}
```

---

## 57. **LassoLars (Least Angle Regression)**

```python
{
    'lassolars__alpha': [0.1, 1.0, 10, 100],
    'lassolars__fit_intercept': [True, False],
    'lassolars__normalize': [True, False]
}
```

---

## 58. **TransformedTargetRegressor**

```python
{
    'transformedtarget__regressor': [LinearRegression(), Ridge()],
    'transformedtarget__transformer': ['passthrough', 'power'],
    'transformedtarget__inverse_transformer': ['log', 'sqrt', 'none']
}
```

---

## 59. **SimpleImputer Regressor**

```python
{
    'simpleimputer__strategy': ['mean', 'median', 'most_frequent', 'constant'],
    'simpleimputer__fill_value': [None, 0, 'missing']
}
```

---

## 60. **Bayesian Gaussian Mixture (Regression with Gaussian Mixture)**

```python
{
    'bgm__n_components': [1, 2, 3],
    'bgm__covariance_type': ['full', 'tied', 'diag
```


', 'spherical'],
'bgm\_\_max\_iter': \[100, 200, 500]
}

````

---