import matplotlib.pyplot as plt

coord = [[1,1],[2,2],[5,10]]

xs,ys = zip(*coord)

plt.plot(xs,ys)
plt.show()