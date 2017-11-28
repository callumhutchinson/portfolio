import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("scraped_data.txt",header=None, delimiter=" ")

timeplot = []
scoreplot = []

for i in range(0,len(data)):
	if (data[2][i]=="AskReddit"):
		scoreplot.append(data[0][i])
		timeplot.append(data[1][i])

plt.scatter(timeplot, scoreplot)
plt.show()


