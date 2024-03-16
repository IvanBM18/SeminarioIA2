import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "./dataset_RegresionLineal.csv"
XTEST = 9.7687

class LinealRegresion:
    
    def __init__(self,ax : plt.Axes) -> None:
        self.weight = 0 #a1
        self.bias = 0 #a0
        self.learnRate = 0.023 #Beta
        self.iterationLimit = 600
        self.error = 0 #J
        self.ax = ax
        
        self.x : np.array
        self.y : np.array
        self.totalElements : int

    def loadDataset(self):
            
        dataset = pd.read_csv(DATASET_PATH)
        self.x = np.array(dataset['x'])
        self.y = np.array(dataset['y'])
        self.totalElements = np.size(self.x)
        
        self.ax.plot(self.x,self.y,'o',color = "yellow",mec = "black")
        
    def fit(self) -> np.array:
        m = self.totalElements
        error_list = []
        for it in range(self.iterationLimit):
            yPredicition = self.hypothesis()
            
            derivateW = (1 / m) * np.dot(self.x.T, yPredicition - self.y) #Forma m,
            derivateB = (1 / m) * np.sum(yPredicition - self.y)
            
            self.weight -= self.learnRate * derivateW
            self.bias -= self.learnRate * derivateB
            
            currError = (1 / (2 * m)) * np.sum(np.power((yPredicition - self.y), 2))
            error_list.append(currError)
            
            print(f"Iteration: {it}\tError: {currError}")
            self.plotPredicition()
        self.plotConvergence(np.array(error_list))
    
    def hypothesis(self):
        return np.dot(self.x,self.weight) + self.bias
    
    def plotPredicition(self,color : str = "yellow",visibility : int = 0.3):
        self.ax.plot(self.x,self.hypothesis(),color= color,alpha = visibility)
        
    def plotConvergence(self,error_list : np.array):
        fig, ax = plt.subplots()
        ax.grid()
        fig.suptitle('Grafica de convergencia')
        ax.plot(error_list,'r')
        ax.set_xlabel('Iteraciones')
        ax.set_ylabel('J (Error)')
        
    def predictData(self,x : int, y : int):
        predicted = self.bias + (self.weight * x)
        print(f'a0={self.bias}, a1={self.weight}')
        print(f'Error actual, J={self.error}')
        print(f'Valor predicho para x={x} es h={predicted}, Valor correcto y={y}')
        self.ax.plot(x,predicted,"o",color = "red",mec = "black")
    
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.grid()
    fig.suptitle('Regresion Lineal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    model = LinealRegresion(ax)
    model.loadDataset()
    model.plotPredicition('red',1)
    model.fit()    
    model.plotPredicition('green',1)
    model.predictData(9.7687,7.5435)
    
    plt.show()
    print("End of program")