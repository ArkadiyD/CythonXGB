import c_xgb_test

c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 10, depth = 5)
c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 10, depth = 10)
c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 5)
c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 10)

c_xgb_test.test_xgb_logistic_regression(n_samples = 10000, n_features = 20, n_estimators = 10, depth = 5)
c_xgb_test.test_xgb_logistic_regression(n_samples = 10000, n_features = 20, n_estimators = 10, depth = 10)
c_xgb_test.test_xgb_logistic_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 5)
c_xgb_test.test_xgb_logistic_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 10)
