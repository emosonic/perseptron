import numpy as np
import math
import random as rm

alfa = 1.0
norma = 5  #скорость обучения

# Активационная функция
def sigmoid(s):
    return 1/(1+math.exp(-alfa*s))

# Первая производная активационной функции (для обратного распределения ошибки)
def sigmoid1(s):
    return sigmoid(s)*(1-sigmoid(s))

# Заполняем обучающую выборку, массив с эталонными значениями
# Столбцы: [bias, X1, X2, X3, X4, y1]
massiv = np.zeros((16,6))  # 16 - кол-во комбинаций из 4 входов
for i in range(16):
    massiv[i,0] = 1 # bias - нейронное смещение, равно 1

# Генерируем ВСЕ возможные комбинации 4 битов (X1, X2, X3, X4)
for i1 in range(2):
    for i2 in range(2):
        for i3 in range(2):
            for i4 in range(2):
                j = int(i1*math.pow(2,3)+i2*math.pow(2,2)+i3*math.pow(2,1)+i4*math.pow(2,0))  # определение строки в массиве
                massiv[j,1] = i1
                massiv[j,2] = i2
                massiv[j,3] = i3
                massiv[j,4] = i4
                massiv[j,5] = ((massiv[j,1]) and (massiv[j,2])) != ((massiv[j,3]) or (massiv[j,4]))  # результат функции

print(massiv)
# Заполняем массив с весовыми коэффициентами нейронов (1 строка - 1 нейрон, число входов + bias)
arrw = np.zeros((3,5)) # 3 - два скрытых + один выходной
for i1 in range(3):
    for i2 in range(5):
        arrw[i1,i2] = 2*rm.random()-1.0


y = np.zeros((3)) # активация нейронов скрытого слоя (bias + 4 входа)
y[0] = 1

print('arrw1', arrw)

for k in range(2000):
    for i in range(16): #каждый пример из массива комбинаций
        # скрытые нейроны
        s1 = 0
        s2 = 0
        
        for j in range(5):
            # вычисление взвешенной суммы скрытых нейронов (веса * входы)
            s1 = s1 + arrw[0,j]*massiv[i,j] 
            s2 = s2 + arrw[1,j]*massiv[i,j]
        
        # активация скрытых нейронов
        y[1] = sigmoid(s1)
        y[2] = sigmoid(s2)
        
        #нахождение взвешенной суммы выходного нейрона (сложение сумм скрытых нейронов * веса)
        s3 = 0
        for j in range(3):
            s3 = s3 + y[j]*arrw[2,j]  # arrw[2,j] - веса выходного нейрона
        
        # активация выходного нейрона
        y3 = sigmoid(s3)
        #вычисление ошибки (дельты)
        delta = massiv[i,5] - y3
        
        print('delta', delta)

        # находим производную от суммы выходного нейрона (насколько нейрон чувствителен к изменениям своей суммы) 
        # и умножаем на дельту, потому что:

        # delta = "на сколько ошиблись"
        # Большая ошибка → нужно сильно менять веса
        # Маленькая ошибка → нужно чуть-чуть менять веса
        delta3 = sigmoid1(s3)*delta
        # в итоге delta3 - это "насколько нужно изменить веса выходного нейрона"

        # sigmoid1(s1) - насколько чувствителен скрытый нейрон 1 к изменениям своей суммы
        # arrw[2,1] - вес связи от скрытого нейрона 1 к выходу (если вес = 0, нейрон не виноват в ошибке)
        # delta3*arrw[2,1] - насколько повлиял на ошибку
        delta1 = sigmoid1(s1)*delta3*arrw[2,1] # delta1 - насколько нужно изменить вес скрытого нейрона 1
        delta2 = sigmoid1(s2)*delta3*arrw[2,2]

        
        # обновляем вес выходного нейрона
        for j in range(3): # 3 входа (bias + 2 скрытых нейрона)
            # к имеющемуся весу прибавляем скорость обучения * градиент (крутизна) выхода * вход (активация скрытого нейрона)
            arrw[2,j] = arrw[2,j] + norma*delta3*y[j]
        
        # обновляем веса от входных нейронов к скрытым
        for j in range(5): # 5 входов (bias, x1-x4) у каждого скрытого нейрона
            # massiv[i,j] - вход
            arrw[0,j] = arrw[0,j] + norma*delta1*massiv[i,j]
        
        for j in range(5):
            arrw[1,j] = arrw[1,j] + norma*delta2*massiv[i,j]
        

# Подсчёт погрешности
epsilon = 0
summa = 0

for i in range(16):
    s1 = 0
    s2 = 0
    
    #прямой проход по все входам, нахождение взвешенных сумм
    for j in range(5):
        s1 = s1 + arrw[0,j]*massiv[i,j]
        s2 = s2 + arrw[1,j]*massiv[i,j]
    
    # активация скрытых нейронов
    y[1] = sigmoid(s1)
    y[2] = sigmoid(s2)
    
    # нахождение взвешенной суммы выходного нейрона (bias + 2 скрытых нейрона)
    s3 = 0
    for j in range(3):
        s3 = s3 + y[j]*arrw[2,j]
    
    # активация выходного нейрона
    y3 = sigmoid(s3)
    # ошибка
    delta = massiv[i,5] - y3
    summa = summa + delta*delta
    x1 = int(massiv[i, 1])
    x2 = int(massiv[i, 2])
    x3 = int(massiv[i, 3])
    x4 = int(massiv[i, 4])
    print(f"{x1}  {x2}  {x3}  {x4}")


summa = math.sqrt(summa/16)

print(summa)