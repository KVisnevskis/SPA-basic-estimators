# SPA Basic Estimators

This repository implements the 6 "basic" estimators that will be used as benchmark comparisons against the proposed LSTM implementation for bending angle estimation. The 6 estimators are:
1. **Pressure-only linear regression with ridge regularisation**
2. **Pressure-only quadratic regression with ridge regularisation**
3. **Pressure + accelerometer linear regression with ridge regularisation**
4. **Pressure + accelerometer quadratic regression with ridge regularisation**
5. **Pressure + accelerometer minimal neural network (MLP)**
6. **Lagged pressure + accelerometer ridge regression**

The data format that the estimators and data loaders present in this repo expect an HDF5 store containing the preprocessed SPA bending datasets (source: https://github.com/KVisnevskis/SPA-data-pre-processing)

The data splits for fitting the estimators is 