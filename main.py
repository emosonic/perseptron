import numpy as np
import math
import random as rm

alfa = 1.0
norma = 5

# Активационная функция
def sigmoid(s):
    return 1/(1+math.exp(-alfa*s))

# Первая производная активационной функции
def sigmoid1(s):
    return sigmoid(s)*(1-sigmoid(s))

# Заполняем обучающую выборку, массив с эталонными значениями
massiv = np.zeros((16,6))
for i in range(16):
    massiv[i,0] = 1

for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                j = int(i1*math.pow(2,3)+i2*math.pow(2,2)+i3*math.pow(2,1)+i4*math.pow(2,0))
                massiv[j,1] = i1
                massiv[j,2] = i2
                massiv[j,3] = i3
                massiv[j,4] = i4
                massiv[j,5] = ((massiv[j,1]) and (massiv[j,2])) != ((massiv[j,3]) or (massiv[j,4]))

print(massiv)
# Заполняем массив с весовыми коэффициентами нейронов (1 строка - 1 нейрон и т.д.)
arrw = np.zeros((3,5))
for i1 in range(3):
    for i2 in range(5):
        arrw[i1,i2] = 2*rm.random()-1.0


y = np.zeros((3))
y[0] = 1

print('arrw1', arrw)

for k in range(2000):
    for i in range(16):
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
        
        for j in range(5):
            s1 = s1 + arrw[0,j]*massiv[i,j]
            s2 = s2 + arrw[1,j]*massiv[i,j]
            s3 = s3 + arrw[2,j]*massiv[i,j]
            s4 = s4 + arrw[3,j]*massiv[i,j]
        
        y[1] = sigmoid(s1)
        y[2] = sigmoid(s2)
        y[3] = sigmoid(s3)
        y[4] = sigmoid(s4)
        
        s5 = 0
        for j in range(3):
            s5 = s5 + y[j]*arrw[4,j]
        
        y5 = sigmoid(s5)
        delta = massiv[i,4] - y5
        
        delta4 = sigmoid1(s4)*delta
        delta1 = sigmoid1(s1)*delta4*arrw[3,1]
        delta2 = sigmoid1(s2)*delta4*arrw[3,2]
        delta3 = sigmoid1(s3)*delta4*arrw[3,3]
        
        for j in range(4):
            arrw[3,j] = arrw[3,j] + norma*delta4*y[j]
        
        for j in range(4):
            arrw[0,j] = arrw[0,j] + norma*delta1*massiv[i,j]
        
        for j in range(4):
            arrw[1,j] = arrw[1,j] + norma*delta2*massiv[i,j]
        
        for j in range(4):
            arrw[2,j] = arrw[2,j] + norma*delta3*massiv[i,j]

# Подсчёт погрешности
epsilon = 0
summa = 0

for i in range(8):
    s1 = 0
    s2 = 0
    s3 = 0
    
    for j in range(4):
        s1 = s1 + arrw[0,j]*massiv[i,j]
        s2 = s2 + arrw[1,j]*massiv[i,j]
        s3 = s3 + arrw[2,j]*massiv[i,j]
    
    y[1] = sigmoid(s1)
    y[2] = sigmoid(s2)
    y[3] = sigmoid(s3)
    
    s4 = 0
    for j in range(4):
        s4 = s4 + y[j]*arrw[3,j]
    
    y4 = sigmoid(s4)
    delta = massiv[i,4] - y4
    summa = summa + delta*delta

summa = math.sqrt(summa/8)
print(summa)