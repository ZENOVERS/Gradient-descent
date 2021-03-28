import numpy as np
import matplotlib.pyplot as plt

data = [[1, 5], [2, 4], [3, 42], [4, 5], [5, 75]]
x = [i[0] for i in data]
y = [i[1] for i in data]
    
plt.figure(figsize=(8, 5))
plt.scatter(x, y)
plt.show()

x_data = np.array(x)
y_data = np.array(y)

a = 0
b = 0
lr = 0.03     #학습률
epochs = 2001  #반복 횟수

for i in range(epochs) :
    y_pred = x_data * a + b
    error = y_data - y_pred
    
    a_diff = -(2/len(x)) * sum(error * x_data) #평균 제곱 오차를 a로 미분
    b_diff = -(2/len(y)) * sum(error)          #평균 제곱 오차를 b로 미분
    
    a -= lr * a_diff  #학습률을 곱해 기존의 a값 업데이트
    b -= lr * b_diff
    
    if i % 100 == 0 :
        print("Epoch = %d, a = %.4f, b = %.4f" %(i, a, b))
       
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()
