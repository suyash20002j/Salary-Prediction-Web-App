import numpy as np 
import pickle

with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]
# country, edlevel, yearscode
X = np.array([["United States", 'Master’s degree', 15 ]])
X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)

salary = regressor.predict(X)