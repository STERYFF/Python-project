import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)

x = data["YearsExperience"]
y = data["Salary"]


def compute_gradient(x, y, w, b):
    w_gradient = (x*(w*x+b-y)).mean()  # 微分結果都有乘二，所以可省略
    b_gradient = ((w*x+b-y)).mean()  # 只讓learning rate影響步伐大小
    return w_gradient, b_gradient


def compute_cost(x, y, w, b):
    y_pred = w*x + b
    cost = pow((y-y_pred), 2)
    cost = cost.sum()/len(x)
    return cost


w = 0
b = 0
learning_rate = 0.001


def gradient_descent(x, y, w_init, b_init, learning_rate, cost_function, gradient_function, run_iter, p_iter=1000):

    c_hist = []
    w_hist = []
    b_hist = []

    w = w_init
    b = b_init

    for i in range(run_iter):
        w_gradient, b_gradient = compute_gradient(x, y, w, b)
        w = w-w_gradient*learning_rate
        b = b-b_gradient*learning_rate
        cost = compute_cost(x, y, w, b)

        w_hist.append(w)
        b_hist.append(b)
        c_hist.append(cost)

        if i % p_iter == 0:
            print(f'Ieration {i:5} : Cost{cost: .4e} , w : {w:.2e} , b : {
                b:.2e} , w_gradient : {w_gradient:.2e} , b_gradient : {b_gradient:.2e}')

    return w, b, w_hist, b_hist, c_hist


w_init = 0
b_init = 0
learning_rate = 0.001
run_iter = 20000

w_final, b_final, w_hist, b_hist, c_hist = gradient_descent(x, y, w_init, b_init, learning_rate,
                                                            compute_cost, compute_gradient, run_iter)

print(f'最終w,b=({w_final:.2f},{b_final:.2f})')

print(f"年資3.5 {w_final*3.5+b_final:.1f}K")
print(f"年資5.9 預測薪水:{w_final*5.9+b_final:.1f}K")

plt.plot(np.arange(0, 100), c_hist[:100])
plt.title("itertion vs cost")
plt.xlabel("itertion")
plt.ylabel("cost")
plt.show()
