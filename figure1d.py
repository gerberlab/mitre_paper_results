import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
plt.ion()

perf = pd.read_csv('performance_results.csv',index_col = 0)

perf = perf.loc[['knat_benchmark','david_benchmark','t1d_benchmark','digiulio_benchmark','bdiet_benchmark']]
perf = perf.drop('group',axis=1)
perf['index'] = perf.index
perf = pd.melt(perf, id_vars=['index'], value_vars=perf.columns[:4])

f,ax = plt.subplots(figsize=(7,3))
sns.barplot(data=perf, x='index', y='value', hue='variable')
ax.legend_.remove()
ax.set_yticks(np.linspace(0.2,1.0,5))
ax.set_xticklabels(['knat','david','t1d','dig','bdiet'])
sns.despine()
f.savefig('figure1d.base.pdf')
f.savefig('figure1d.base.eps')
