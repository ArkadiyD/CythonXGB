# CythonXGB is fast one-sample prediction for XGBoost for usage with Cython

In some cases it is required to make online predictions, particularly, with trained XGBoost model. This project is designed to use trained XGBoost model for online predictions for Cython arrays many times faster than with usual XGBoost Scikit-Learn API.

Files description:
* `c_xgb.cpp` - classes for predictions for XGBoost models written in C++
* `c_xgb.h` - header for `c_xgb.cpp`
* `json.cpp`, `json.hpp` - JSON parser taken from https://github.com/nbsdx/SimpleJSON/ and modified
* `c_xgb_test.pyx` - Cython file with tests for CythonXGB
* `setup.py` - setup file for tests compilation
* `c_xgb_test.py` - Python file which runs tests

# Requirements
* Python: 2.7.12
* Cython: 0.25.2
* Numpy: 1.13.0
* Scikit-Learn: 0.18.1
* XGBoost: up-to-day version built from source

# Installation and setup
0. **Note**: current version of XGBoost still has problems with precision of floats in dumped models (https://github.com/dmlc/xgboost/issues/1204).
If your task is sensitive to high floating point precision in features, you can avoid differences in predictions by XGBoost and CythonXGB which may occur because of not enough precision in XGBoost dumps, please, change XGBoost source according to https://github.com/dmlc/xgboost/issues/1204#issuecomment-219892846 and rebuild it.
1. `mkdir trees`
2. Compile Cython module: `python setup.py build_ext --inplace`
3. To run tests: `python c_xgb_test.py`

# CythonXGB usage in Cython files:
   
1. Extern class from CPP file:
```python
cdef extern from "c_xgb.cpp":
	cdef cppclass CXgboost:
		CXgboost()
		CXgboost(int depth, int n_features, int n_trees_ , int objective_, double base_score_)
		double predict(double *features)
```

2. Train your XGBoost model and dump it:
```python
tree_data = booster.get_dump(dump_format='json')
	cdef i
	for i in xrange(len(tree_data)):
		f = open("trees/tree_%d.json" % i, 'w')
		f.write(tree_data[i])
		f.close()
```

3. After training you can create an instance of CXgboost model:
```python
cdef CXgboost model_c = CXgboost(depth, n_features, n_estimators, 0, base_score)
```

4. To make predictions for one sample:
```python
preds_c_xgb = model_c.predict(x_cy)
```
`x_cy` should be a float cython array with current sample features

# Tests
For tests were generated classification and regression datasets with Scikit-Library, they contain 10000 samples with 20 features. 
Predicitons were made for each sample 10 times

*Mean_time_XGBoost* is mean time for prediction for one sample by Scikit-Learn wrapper for XGBoost
*Mean_time_CythonXGB* is mean time for prediction for one sample by CythonXGB

*Acceleration* is calculated as *Mean_time_XGBoost* / *Mean_time_CythonXGB*

Tests are performed on laptop with Intel® Core™ i5-4210U CPU, Ubuntu 16.04

| n_estimators        | max_depth           | objective  | acceleration, times |
| ------------- |:-------------:| -----:| -----:|
| 10      | 5 | "reg:linear" | 461 |
| 10      | 10 | "reg:linear"    | 188   |
| 100      | 5 | "reg:linear" | 49 |
| 100      | 10 | "reg:linear"    | 14   |
| 10 | 5      |    "binary:logistic" | 730 |
| 10 | 10      |    "binary:logistic" | 343 |
| 100 | 5      |    "binary:logistic" | 84 |
| 100 | 10      |    "binary:logistic" | 24 |
