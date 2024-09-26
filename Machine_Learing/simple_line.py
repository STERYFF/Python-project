from cProfile import label
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact

url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)

# y=w*x+b
x = data["YearsExperience"]
y = data["Salary"]

w = 0
b = 0


def plot_pred(w, b):
    y_pred = w*x + b
    plt.plot(x, y_pred, color="blue", label="pred")  # 畫出指定的直線
    plt.scatter(x, y, marker="x", color="red",
                label="true_info")  # 點出給定的 x y 座標圖
    plt.title("YearsExperience-Salary")
    plt.xlabel("YearsExperience")
    plt.ylabel("Salary(thousand)")
    plt.xlim([0, 12])
    plt.ylim([-60, 140])
    plt.legend()
    plt.show()


interact(plot_pred, w=(-100, 100, 1), b=(-100, 100, 1))
