"""Compare saved full-chain runtimes; each dot is one completed trial."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
comparisons = json.loads((ROOT/'batching_comparison_1m.json').read_text())
floors = json.loads((ROOT/'in_floor_comparison_extended.json').read_text())
h = 1e-4
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 1.4]})
fig.subplots_adjust(left=.065, right=.98, bottom=.20, top=.72, wspace=.27)
fig.text(.065,.94,'Runtime Comparisons',fontsize=24,weight='bold',color='#17334a')
fig.text(.065,.87,'10D simplex  |  h = 10⁻⁴  |  72,924 steps per output',fontsize=12,color='#52616b')

def points_and_medians(ax, groups, labels):
    for i, group in enumerate(groups):
        values=np.asarray([r['seconds'] for r in group])
        if not len(values):
            raise ValueError(f'No trials for {labels[i]}')
        median=float(np.median(values))
        ax.scatter(i+np.linspace(-.08,.08,len(values)),values,s=25,color='#8fa3ad',alpha=.8,zorder=3)
        ax.plot([i-.22,i+.22],[median,median],lw=3,color='#007f83',zorder=4)
    ax.set_xticks(range(len(labels)),labels)
    ax.set_xlim(-.5,len(labels)-.5)
    ax.set_ylabel('Runtime (s)')
    ax.grid(axis='y',alpha=.18,zorder=0)

ax=axes[0]
ax.set_title('Optimizations · 5 trials each',loc='left',fontsize=13,pad=14)
groups=[[r for r in comparisons['runs'] if r['h']==h and r['variant']==v]
        for v in ['scalar','prefetch_out','batch_in']]
points_and_medians(ax,groups,['Original','OUT prefetch\nonly','IN batching\nonly'])
ax.set_yscale('log')
ax.set_ylim(1.3,max(40,max(r['seconds'] for g in groups for r in g)*1.1))
ax.set_yticks([2,5,10,20,40]);ax.yaxis.set_major_formatter(ScalarFormatter());ax.minorticks_off()
ax.set_ylabel('Runtime (s, log scale)')
ax=axes[1]
ax.set_title('IN batch floor · 10 trials each',loc='left',fontsize=13,pad=14)
groups=[[r for r in floors['runs'] if r['h']==h and r['floor']==f] for f in floors['floors']]
points_and_medians(ax,groups,[str(f) for f in floors['floors']])
ax.set_xlabel('Floor (after first rejection)')
ax.set_ylim(0,max(r['seconds'] for g in groups for r in g)*1.1)
fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color='#8fa3ad',label='Trial'),
                    Line2D([],[],color='#007f83',lw=3,label='Median')],
           loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False)
fig.text(.065,.025,'Separate experiments; IN batching changes random paths. Timings exclude plotting.',fontsize=9,color='#52616b')
output=ROOT/'advisor_optimization_summary.png'
fig.savefig(output,dpi=200,facecolor='white')
plt.close(fig)
print(output)
