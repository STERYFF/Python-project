from turtle import color
from matplotlib import projections
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)

# y=w*x+b
x = data["YearsExperience"]
y = data["Salary"]

w = 10
b = 0

# y_pred = w*x + b
# cost = pow((y-y_pred), 2)
# print(cost.sum()/len(x))


def compute_cost(x, y, w, b):
    y_pred = w*x + b
    cost = pow((y-y_pred), 2)
    cost = cost.sum()/len(x)
    print(cost)

    return cost


# compute_cost(x, y, 10, 10)

# b=0 w=-100~100 cost 會是多少

# costs = []
# for w in range(-100, 101):
#     cost = compute_cost(x, y, w, 0)
#     costs.append(cost)

# plt.plot(range(-100, 101), costs)
# plt.title("cost function b=0 w=-100~100")
# plt.xlabel("w")
# plt.ylabel("cost")
# plt.show()

# w=-100~100 b=-100~100 的 cost
ws = np.arange(-100, 101)
bs = np.arange(-100, 101)
costs = np.zeros((201, 201))

i = 0
for w in ws:
    j = 0
    for b in bs:
        cost = compute_cost(x, y, w, b)
        costs[i, j] = cost
        j = j+1
    i = i+1

# print(costs)

plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")
ax.view_init(45, -120)
b_grid, w_grid = np.meshgrid(bs, ws)
ax.plot_surface(w_grid, b_grid, costs, cmap="Spectral_r", alpha=0.7)
ax.plot_wireframe(w_grid, b_grid, costs, color="black", alpha=0.1)
ax.set_title("w , b -cost")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("cost")

print(np.min(costs))
w_index, b_index = np.where(costs == np.min(costs))
print(f'當w={ws[w_index]},且b= {bs[b_index]} cost最小 是{costs[w_index, b_index]}')
ax.scatter(ws[w_index], bs[b_index],
           costs[w_index, b_index], color="red", s=40)

plt.show()
