import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
plt.ion()

all_perf = pd.read_csv('performance_results.csv',index_col = 0)

perf = all_perf.loc[['knat_benchmark','david_benchmark','t1d_benchmark','digiulio_benchmark','bdiet_benchmark']]
perf = perf.drop('group',axis=1)
perf['index'] = perf.index
perf = pd.melt(perf, id_vars=['index'], value_vars=perf.columns[:4])

f,ax = plt.subplots(figsize=(7,3))
sns.barplot(data=perf, x='index', y='value', hue='variable', hue_order=['mitre_ensemble_f1', 'mitre_point_f1', 'l1_f1', 'rf_f1'])
ax.set_yticks(np.linspace(0.2,1.0,5))
ax.set_ylabel('F1 score')
ax.set_ylim(0.,1.)
ax.set_xticklabels(['Karelia','David','Kostic','DiGiulio','Bokulich'])
ax.set_xlabel('')
sns.despine()
f.savefig('figure1d.base.pdf')
f.savefig('figure1d.base.eps')
ax.legend_.remove()
f.savefig('figure1d.base.nolegend.pdf')
f.savefig('figure1d.base.nolegend.eps')

# Make the supplementary figure showing the David results with
# and without Bacteroides normalization
perf = all_perf.loc[['david_benchmark','david_benchmark.bacteroides']]
perf = perf.drop('group',axis=1)
perf['index'] = perf.index
perf = pd.melt(perf, id_vars=['index'], value_vars=perf.columns[:4])

rename = lambda f: {'mitre_ensemble_f1': 'MITRE ensemble',
                    'mitre_point_f1': 'MITRE point estimate',
                    'rf_f1': 'Random forest',
                    'l1_f1': 'L1-regularized logistic regression'}.get(f)
perf['variable'] = perf.variable.apply(rename)

f,ax = plt.subplots(1,2,figsize=(7,3))
ax[1].axis('off')
ax = ax[0]
sns.barplot(ax=ax,data=perf, x='index', y='value', hue='variable')
#ax.legend_.remove()
ax.set_yticks(np.linspace(0.2,1.0,5))
ax.set_ylim(0.,1.)
ax.set_ylabel('F1 score')
ax.set_xlabel('')
ax.legend(bbox_to_anchor=(0.95,0.65))
ax.set_xticklabels(['standard','renormalized'])
sns.despine()
f.tight_layout()
f.savefig('bacteroides_figure.base.pdf')
f.savefig('bacteroides_figure.base.eps')

