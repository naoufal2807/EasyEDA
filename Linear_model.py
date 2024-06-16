from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, TweedieRegressor,Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.datasets import make_regression
import pandas as pd 
import numpy as np

#  Inherited class 
class LinearModel():

    def __init__(self, model) -> None:

        self.model = model
    
    def train_test(self, X, y) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        MSE = mean_squared_error(y_test, y_pred)
        MAE = mean_absolute_error(y_test, y_pred)
        MAPE = mean_absolute_percentage_error(y_test, y_pred)

        evaluation = pd.DataFrame({'MSE': [MSE]
                                   ,'MAE': [MAE]      # metrics are in scalar format so it wise adding []
                                    ,'MAPE': [MAPE]})
        print(evaluation)

#inheriting class

class RidgeModel(LinearModel):
    """"
    in Ridge : There are some parameters 
    solver : 'auto' , 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag', 'lbfgs'
   
    """

    def __init__(self, solver= 'auto',) -> None:
        
        model = Ridge(solver=solver)
        super().__init__(model)
        self.solver = solver


    
X,y = make_regression(n_samples=1200, n_features= 12, n_targets= 1)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_ = RidgeModel(solver='cholesky')
model_.train(X_train, y_train)

model_.evaluate(X_test, y_test)









