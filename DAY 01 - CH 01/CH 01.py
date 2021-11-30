# %% IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

from prml.preprocess import PolynomialFeature
from prml.linear import (
    LinearRegression,
    RidgeRegression,
    BayesianRegression
)

np.random.seed(1234)
# %% DEFINE THE FUNCTIONS
def create_dummy_data(func, sample_size, std):
    # create data, x, that has values between 0 and 1
    x = np.linspace(0, 1, sample_size)
    # apply the pattern, func, to the feature values, x,
    # and add random noise to it
    t = func(x) + np.random.normal(scale=std,
                                   size=x.shape)
    return x, t


def data_pattern(x):
    return np.sin(2 * np.pi * x)


# %% GENERATE TRAINING AND TEST DATA
x_train, y_train = create_dummy_data(data_pattern,
                                     10, 0.25)

x_test = np.linspace(0, 1, 100)
y_test = data_pattern(x_test)
# %% PLOT THE DATA
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

# PLOT THE SIN FUNCTION (x_test and y_test)
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")
plt.legend()
plt.show()
# %% Fitting a model of degree 0
degree = 0
# Get Polynomial feature of degree 0
feature = PolynomialFeature(degree)
x_train_0 = feature.transform(x_train)
x_test_0 = feature.transform(x_test)

# Define regression model
model = LinearRegression()
# Let the model find the pattern of the data 
#   (fittin the model to the data)
model.fit(x_train_0, y_train)

# Lets predict target values for the test data
y_0 = model.predict(x_test_0)
# %% Plot the train and test data, and regression line

# plot train data
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

# plot test data with their original target values
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

# plot our prediction line (model perfromance)
plt.plot(x_test, y_0, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
# %% Fitting a model of degree 1
degree = 1
# Get Polynomial feature of degree 1
feature = PolynomialFeature(degree)
x_train_1 = feature.transform(x_train)
x_test_1 = feature.transform(x_test)

# Define regression model
model = LinearRegression()
# Let the model find the pattern of the data 
#   (fittin the model to the data)
model.fit(x_train_1, y_train)

# Lets predict target values for the test data
y_1 = model.predict(x_test_1)
# %% Plot the train and test data, and regression line

# plot train data
plt.scatter(x_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="Training Data")

# plot test data with their original target values
plt.plot(x_test, y_test, c="green", label="$\sin(2\pi x)$")

# plot our prediction line (model perfromance)
plt.plot(x_test, y_1, c="red", label="fitting")

plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()
#%% CREATE THE RMSE FUNCTION 
def rmse(predicted_val, true_val):
    return np.sqrt(
        np.mean(
            np.square(
                predicted_val - true_val
                )
            )
        )
#%% PRINT RMSE VALUES OF THE TWO MODELS
rmse_0 = rmse(y_0, y_test)
rmse_1 = rmse(y_1, y_test)
print("The RMSE of a model of degree 0 is", 
      rmse_0)
print("The RMSE of a model of degree 1 is", 
      rmse_1)
#%% RMSE OF 10 MODELS
train_errors = []
test_errors = []

for i in range(10):
    feature = PolynomialFeature(i)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model prediction on test set
    y = model.predict(X_test)
    # Model prediction on train set
    z = model.predict(X_train)
    
    rmse_train = rmse(z, y_train)
    rmse_test = rmse(y, y_test)

    train_errors.append(rmse_train)
    test_errors.append(rmse_test)
#%% PLOT train_errors and test_errors    
plt.plot(train_errors, 'o-', 
         mfc="none", mec="b", 
         ms=10, c="b", label="Training")
plt.plot(test_errors, 'o-', 
         mfc="none", mec="r", 
         ms=10, c="r", label="Test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.show()
#%%
# features of degree 9
feature = PolynomialFeature(9)
# training set of degree 9
X_train = feature.transform(x_train)
# test set of degree 9
X_test = feature.transform(x_test)
#%%
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
y_lin_reg = lin_reg_model.predict(X_test)

plt.scatter(x_train, y_train, facecolor="none", 
            edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y_lin_reg, c="r", label="fitting")
plt.ylim(-1.5, 1.5)
plt.legend()
plt.annotate("M=9", xy=(-0.15, 1))
plt.show()
#%%
rid_reg_model = RidgeRegression(alpha=1e-3)
rid_reg_model.fit(X_train, y_train)
y_rid_reg = rid_reg_model.predict(X_test)

plt.scatter(x_train, y_train, facecolor="none", 
            edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y_rid_reg, c="r", label="fitting")
plt.ylim(-1.5, 1.5)
plt.legend()
plt.annotate("M=9", xy=(-0.15, 1))
plt.show()
#%%
rmse_lin_reg = rmse(y_lin_reg, y_test)
rmse_rid_reg = rmse(y_rid_reg, y_test)
print("The RMSE of a linear regression model is",
      rmse_lin_reg)
print("The RMSE of a ridge regrssion model is",
      rmse_rid_reg)
#%%
print(rid_reg_model.w)
print()
print(lin_reg_model.w)
#%%
bas_model = BayesianRegression(alpha=2e-3, 
                               beta=2)
bas_model.fit(X_train, y_train)

bas_y, y_std = bas_model.predict(X_test, 
                                 return_std=True)
#%%
plt.scatter(x_train, y_train, facecolor="none", 
            edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, bas_y, c="r", label="mean")
plt.fill_between(x_test, bas_y - y_std, bas_y + y_std, 
                 color="pink", label="std.", alpha=0.5)
plt.xlim(-0.1, 1.1)
plt.ylim(-1.5, 1.5)
plt.annotate("M=9", xy=(0.8, 1))
plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0.)
plt.show()