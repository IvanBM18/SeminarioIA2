import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATASET_PATH = "./dataset_RegresionLineal.csv"
XTEST = 9.7687

dataset = pd.read_csv(DATASET_PATH)
x = np.array(dataset['x']).reshape(-1,1)
y = np.array(dataset['y'])

plt.plot(x,y,'o',color = "yellow",mec = "black")
plt.xlabel('X')
plt.ylabel('Y')

model = LinearRegression()
model.fit(x,y)  #Entrenamiento
predictedY = model.predict(x)  #Predicción
plt.plot(x,predictedY,"g")

predictedY = model.predict([[XTEST]])  #Predicción
plt.plot(XTEST,predictedY,"ro")
print(f'a0 = {model.intercept_}, a1 = {model.coef_}')
print(f'Valor predicho para el dato de prueba  X={XTEST}, es {predictedY[0]}')

plt.title('Regresión Lineal con SKLearn')
plt.show()