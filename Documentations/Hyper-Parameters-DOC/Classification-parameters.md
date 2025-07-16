## 1. **Logistic Regression**

```python
{
    'logreg__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'logreg__C': [0.01, 0.1, 1.0, 10],
    'logreg__solver': ['lbfgs', 'liblinear', 'saga'],
    'logreg__max_iter': [100, 200, 500]
}
```

---

## 2. **K-Nearest Neighbors (KNN)**

```python
{
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__p': [1, 2],  # 1=Manhattan, 2=Euclidean
    'knn__leaf_size': [30, 50, 100]
}
```

---

## 3. **Decision Tree Classifier**

```python
{
    'tree__max_depth': [None, 5, 10, 20, 50],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 2, 4],
    'tree__max_features': ['auto', 'sqrt', 'log2'],
    'tree__criterion': ['gini', 'entropy']
}
```

---

## 4. **Random Forest Classifier**

```python
{
    'rf__n_estimators': [100, 200, 500],
    'rf__max_depth': [None, 10, 20, 50],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['auto', 'sqrt', 'log2'],
    'rf__bootstrap': [True, False]
}
```

---

## 5. **Gradient Boosting Classifier**

```python
{
    'gb__n_estimators': [100, 200, 500],
    'gb__learning_rate': [0.01, 0.1, 0.5],
    'gb__max_depth': [3, 5, 10],
    'gb__min_samples_split': [2, 5, 10],
    'gb__min_samples_leaf': [1, 2, 4],
    'gb__subsample': [0.5, 0.8, 1.0]
}
```

---

## 6. **XGBoost Classifier (xgboost.XGBClassifier)**

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

## 7. **Support Vector Machine (SVM) Classifier**

```python
{
    'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto'],
    'svm__degree': [2, 3, 4],
    'svm__probability': [True, False]
}
```

---

## 8. **AdaBoost Classifier**

```python
{
    'ada__n_estimators': [50, 100, 200],
    'ada__learning_rate': [0.01, 0.1, 1.0],
    'ada__algorithm': ['SAMME', 'SAMME.R']
}
```

---

## 9. **Bagging Classifier**

```python
{
    'bag__n_estimators': [10, 50, 100],
    'bag__max_samples': [0.5, 0.7, 1.0],
    'bag__max_features': [0.5, 0.7, 1.0],
    'bag__bootstrap': [True, False]
}
```

---

## 10. **Extra Trees Classifier**

```python
{
    'et__n_estimators': [100, 200, 500],
    'et__max_depth': [None, 10, 20, 30],
    'et__min_samples_split': [2, 5, 10],
    'et__min_samples_leaf': [1, 2, 4],
    'et__max_features': ['auto', 'sqrt', 'log2'],
    'et__bootstrap': [True, False]
}
```

---

## 11. **K-Nearest Neighbors Classifier (Enhanced)**

```python
{
    'knn__n_neighbors': [3, 5, 7, 9, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__p': [1, 2],
    'knn__leaf_size': [30, 50, 100],
    'knn__metric': ['minkowski', 'euclidean', 'manhattan']
}
```

---

## 12. **LightGBM Classifier**

```python
{
    'lgbm__num_leaves': [31, 50, 100],
    'lgbm__max_depth': [-1, 5, 10],
    'lgbm__learning_rate': [0.01, 0.05, 0.1],
    'lgbm__n_estimators': [100, 200, 500],
    'lgbm__subsample': [0.6, 0.8, 1.0],
    'lgbm__colsample_bytree': [0.6, 0.8, 1.0],
    'lgbm__boosting_type': ['gbdt', 'dart', 'goss']
}
```

---

## 13. **CatBoost Classifier**

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

## 14. **Naive Bayes (GaussianNB)**

```python
{
    'nb__var_smoothing': [1e-9, 1e-8, 1e-7]
}
```

---

## 15. **Neural Network Classifier (MLPClassifier)**

```python
{
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'mlp__activation': ['relu', 'tanh', 'logistic'],
    'mlp__solver': ['adam', 'lbfgs'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__max_iter': [100, 200, 500]
}
```

---

## 16. **PassiveAggressive Classifier**

```python
{
    'pa__C': [0.01, 0.1, 1.0, 10],
    'pa__max_iter': [50, 100, 200],
    'pa__tol': [1e-3, 1e-4, 1e-5],
    'pa__loss': ['hinge', 'log']
}
```

---

## 17. **RidgeClassifier**

```python
{
    'ridge__alpha': [0.01, 0.1, 1.0, 10],
    'ridge__fit_intercept': [True, False],
    'ridge__normalize': [True, False]
}
```

---

## 18. **Linear Discriminant Analysis (LDA)**

```python
{
    'lda__solver': ['lsqr', 'eigen'],
    'lda__priors': [None, 'uniform'],
    'lda__shrinkage': [None, 'auto', 0.1, 0.5],
    'lda__n_components': [1, 2]
```


}

---

## 19. **Quadratic Discriminant Analysis (QDA)**

```python
{
    'qda__priors': [None, 'uniform'],
    'qda__reg_param': [0.0, 0.1, 0.5],
    'qda__tol': [1e-4, 1e-3, 1e-2]
}
```

---

## 20. **Ridge Classifier (with regularization)**

```python
{
    'ridgeclf__alpha': [0.01, 0.1, 1.0, 10],
    'ridgeclf__fit_intercept': [True, False],
    'ridgeclf__normalize': [True, False]
}
```

---

## 21. **Nearest Centroid Classifier**

```python
{
    'centroid__metric': ['euclidean', 'manhattan', 'chebyshev'],
    'centroid__shrink_threshold': [None, 0.1, 0.5]
}
```

---

## 22. **Multinomial Naive Bayes (MultinomialNB)**

```python
{
    'mnb__alpha': [0.1, 0.5, 1.0, 2.0],
    'mnb__fit_prior': [True, False]
}
```

---

## 23. **Bernoulli Naive Bayes (BernoulliNB)**

```python
{
    'bnb__alpha': [0.1, 0.5, 1.0, 2.0],
    'bnb__fit_prior': [True, False],
    'bnb__binarize': [0.0, 0.5, 1.0]
}
```

---

## 24. **SVC with RBF Kernel (Support Vector Classification)**

```python
{
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['rbf', 'poly', 'sigmoid'],
    'svc__degree': [2, 3, 4],
    'svc__gamma': ['scale', 'auto'],
    'svc__probability': [True, False]
}
```

---

## 25. **Gaussian Mixture Model (GMM)**

```python
{
    'gmm__n_components': [1, 2, 3, 4],
    'gmm__covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'gmm__max_iter': [100, 200, 500],
    'gmm__tol': [1e-3, 1e-4, 1e-5]
}
```

---

## 26. **LinearSVC (Support Vector Classifier)**

```python
{
    'linsvc__C': [0.1, 1, 10],
    'linsvc__loss': ['hinge', 'squared_hinge'],
    'linsvc__penalty': ['l2', 'none'],
    'linsvc__dual': [True, False]
}
```

---

## 27. **Dummy Classifier**

```python
{
    'dummy__strategy': ['stratified', 'most_frequent', 'uniform', 'constant'],
    'dummy__constant': [0, 1]
}
```

---

## 28. **RBF Network (Radial Basis Function Network)**

```python
{
    'rbf__gamma': [0.01, 0.1, 1.0, 10],
    'rbf__n_neighbors': [5, 10, 20],
    'rbf__max_iter': [100, 200, 500]
}
```

---

## 29. **Stochastic Gradient Descent (SGDClassifier)**

```python
{
    'sgd__loss': ['hinge', 'log', 'squared_hinge', 'perceptron'],
    'sgd__penalty': ['l2', 'l1', 'elasticnet'],
    'sgd__alpha': [0.0001, 0.001, 0.01],
    'sgd__max_iter': [1000, 2000, 3000],
    'sgd__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'sgd__eta0': [0.01, 0.1, 1.0]
}
```

---

## 30. **Perceptron**

```python
{
    'perceptron__penalty': ['l2', 'l1', 'elasticnet'],
    'perceptron__alpha': [0.0001, 0.001, 0.01],
    'perceptron__fit_intercept': [True, False],
    'perceptron__max_iter': [1000, 2000, 3000],
    'perceptron__eta0': [0.1, 0.5, 1.0]
}
```

---

## 31. **Neural Network Classifier (MLPClassifier Enhanced)**

```python
{
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (200,)],
    'mlp__activation': ['relu', 'tanh', 'logistic'],
    'mlp__solver': ['adam', 'lbfgs', 'sgd'],
    'mlp__alpha': [0.0001, 0.001, 0.01, 0.1],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__max_iter': [100, 200, 500, 1000],
    'mlp__early_stopping': [True, False]
}
```

---

## 32. **Tweedie Generalized Linear Model (TweedieGLM)**

```python
{
    'tweedie__power': [1, 1.5, 2, 2.5],
    'tweedie__alpha': [0.1, 0.5, 1.0],
    'tweedie__link': ['log', 'identity', 'logit']
}
```

---

## 33. **Log-Linear Model (Multinomial Logistic Regression)**

```python
{
    'loglinear__solver': ['newton-cg', 'lbfgs', 'saga'],
    'loglinear__multi_class': ['ovr', 'multinomial'],
    'loglinear__C': [0.1, 1.0, 10],
    'loglinear__max_iter': [100, 200, 500]
}
```

---

## 34. **QDA (Quadratic Discriminant Analysis Enhanced)**

```python
{
    'qda__priors': [None, 'uniform'],
    'qda__reg_param': [0.0, 0.1, 0.5],
    'qda__tol': [1e-4, 1e-3, 1e-2],
    'qda__covariance_type': ['full', 'tied']
}
```

---

## 35. **Fuzzy Logic Classifier (Fuzzy C-Means)**

```python
{
    'fcm__m': [1.5, 2.0, 2.5],
    'fcm__max_iter': [100, 200, 500],
    'fcm__tolerance': [1e-4, 1e-3, 1e-2]
}
```

---

## 36. **Logistic Regression (with regularization)**

```python
{
    'logreg__penalty': ['l2', 'l1'],
    'logreg__C': [0.1, 1, 10],
    'logreg__solver': ['liblinear', 'lbfgs', 'saga'],
    'logreg__max_iter': [100, 200, 300]
}
```

---

## 37. **K-Nearest Neighbors (KNN Classifier)**

```python
{
    'knn__n_neighbors': [3, 5, 7, 9, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__leaf_size': [20, 30, 40]
}
```

---

## 38. **GradientBoostingClassifier**

```python
{
    'gbc__n_estimators': [100, 200, 500],
    'gbc__learning_rate': [0.01, 0.1, 0.5],
    'gbc__max_depth': [3, 5, 7],
    'gbc__min_samples_split': [2, 5, 10],
    'gbc__min_samples_leaf': [1, 2, 4],
    'gbc__subsample': [0.7, 0.8, 1.0]
}
```

---

## 39. **LightGBM Classifier (lightgbm.LGBMClassifier)**

```python
{
    'lgbm__n_estimators': [100, 200, 500],
    'lgbm__learning_rate': [0.01, 0.1, 0.2],
    'lgbm__num_leaves': [31, 50, 100],
    'lgbm__max_depth': [-1, 10, 20],
    'lgbm__subsample': [0.7, 0.8, 1.0],
    'lgbm__colsample_bytree': [0.5, 0.7, 1.0]
}
```

---

## 40. **CatBoost Classifier**

```python
{
    'catboost__iterations': [100, 200, 500],
    'catboost__learning_rate': [0.01, 0.05, 0.1],
    'catboost__depth': [6, 8, 10],
    'catboost__l2_leaf_reg': [1, 3, 5],
    'catboost__bagging_temperature': [0.0, 0.5, 1.0]
}
```

---

## 41. **XGBoost Classifier (xgboost.XGBClassifier)**

```python
{
    'xgb__n_estimators': [100, 200, 500],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__max_depth': [3, 6, 10],
    'xgb__subsample': [0.5, 0.8, 1.0],
    'xgb__colsample_bytree': [0.5, 0.8, 1.0],
    'xgb__reg_alpha': [0, 0.1, 1.0],
    'xgb__reg_lambda': [1.0, 1.5, 2.0]
}
```

---

## 42. **Neural Network Classifier (MLPClassifier)**

```python
{
    'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (200,)],
    'mlp__activation': ['relu', 'tanh', 'logistic'],
    'mlp__solver': ['adam', 'lbfgs', 'sgd'],
    'mlp__alpha': [0.0001, 0.001, 0.01],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__max_iter': [100, 200, 500],
    'mlp__early_stopping': [True, False]
}
```

---

## 43. **Random Forest Classifier**

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

## 44. **AdaBoost Classifier**

```python
{
    'ada__n_estimators': [50, 100, 200],
    'ada__learning_rate': [0.01, 0.1, 1.0],
    'ada__loss': ['linear', 'square', 'exponential']
}
```

---

## 45. **Bagging Classifier**

```python
{
    'bag__n_estimators': [10, 50, 100],
    'bag__max_samples': [0.5, 0.7, 1.0],
    'bag__max_features': [0.5, 0.7, 1.0]
}
```

---

## 46. **Extra Trees Classifier**

```python
{
    'et__n_estimators': [100, 200, 500],
    'et__max_depth': [None, 10, 20, 30],
    'et__min_samples_split': [2, 5, 10],
    'et__min_samples_leaf': [1, 2, 4],
    'et__max_features': ['auto', 'sqrt', 'log2']
}
```

---

## 47. **Decision Tree Classifier**

```python
{
    'tree__max_depth': [None, 5, 10, 20, 50],
    'tree__min_samples_split': [2, 5, 10],
    'tree__min_samples_leaf': [1, 2, 4],
    'tree__criterion': ['gini', 'entropy']
}
```

---

## 48. **KNeighborsClassifier with Metrics (KNN)**

```python
{
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__metric': ['minkowski', 'euclidean', 'manhattan']
}
```

---

## 49. **Linear Discriminant Analysis (LDA)**

```python
{
    'lda__solver': ['svd', 'lsqr', 'eigen'],
    'lda__shrinkage': ['auto', None],
    'lda__priors': [None, 'uniform'],
    'lda__n_components': [None, 1, 2]
}
```

---

## 50. **Multinomial Logistic Regression (Softmax Regression)**

```python
{
    'multilogreg__solver': ['newton-cg', 'lbfgs', 'saga'],
    'multilogreg__multi_class': ['ovr', 'multinomial'],
    'multilogreg__C': [0.1, 1.0, 10],
    'multilogreg__max_iter': [100, 200, 500]
}
```

---
