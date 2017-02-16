#!/bin/python
# import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # HW1: cluster_num vs. inertia
    x = range(50, 501, 50)
    y = [726, 641, 592, 563, 540, 524, 508, 500, 490, 480]
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.title("K-means Clustering Parameter Selection")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia (in Millions)")
    plt.show()
