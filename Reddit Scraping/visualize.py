'''
Format:
0                    1              2            3                4
Net Votes,Hours since submitted,Subreddit,Upvote Ratio,Num users on subreddit

Pandas[col][row]
'''

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("scraped_data.txt",header=None, delimiter=" ")

timeplot = []
scoreplot = []

plt.hist(data[2])


