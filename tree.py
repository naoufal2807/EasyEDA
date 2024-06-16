from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

class Forest :

    def __init__(self, n_estimators=2):
    
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=self.n_estimators)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        evaluation = pd.DataFrame({'accuracy': [accuracy]
                                   ,'recall': [recall]      # metrics are in scalar format so it wise adding []
                                    ,'precision': [precision]})
        print(evaluation)

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Convert to DataFrame (optional, but convenient for handling data)
df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
df['target'] = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Model_1 = Forest(n_estimators=7)
Model_1.train(X_train, y_train)
Model_1.evaluate(X_test, y_test)




    




