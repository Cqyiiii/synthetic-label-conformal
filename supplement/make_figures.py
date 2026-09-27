"""Standalone scientific figures from reviewed CSVs; embedded by report builder."""
import os
from pathlib import Path
import common
os.environ['MPLCONFIGDIR']=str(common.ROOT/'outputs/.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=common.ROOT; A=ROOT/'outputs/analysis'; DEST=ROOT/'figures'; DEST.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#27313b','text.color':'#27313b','axes.edgecolor':'#afb8c2',
                     'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none'})
COLORS=['#286b9d','#bc7629','#947846','#766195']
DATASETS=['bike','diamond','SGEMM','Transcoding']; NAMES={'bike':'Bike','diamond':'Diamonds','SGEMM':'SGEMM','Transcoding':'Transcoding'}
RULES=['raw_pinball','calibrated_pinball','coverage_beta0','coverage_beta1']
LABELS=['Raw pinball','Calibrated pinball','Coverage beta=0','Coverage beta=1']


def save(fig,name):
    fig.savefig(DEST/f'{name}.svg',bbox_inches='tight')
    fig.savefig(DEST/f'{name}.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


paired=pd.read_csv(A/'tabular_paired.csv')
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
for ax,ds in zip(axes.flat,DATASETS):
    rows=paired[(paired.dataset==ds)&(paired.reference=='Fixed-PPI')&(paired.metric=='group_raw_k30')].set_index('method').loc[RULES]
    for i,(name,row) in enumerate(rows.iterrows()):
        ax.errorbar(row.difference*1000,3-i,xerr=[[max(0,row.difference-row.ci95_low)*1000],[max(0,row.ci95_high-row.difference)*1000]],fmt='o',color=COLORS[i],capsize=4,markersize=6,lw=1.6)
    ax.axvline(0,color='#6f7780',lw=1,ls='--'); ax.set_title(NAMES[ds],loc='left',fontweight='bold')
    ax.set_yticks([3,2,1,0],LABELS); ax.grid(axis='x',alpha=.16); ax.set_ylim(-.6,3.6)
for ax in axes[1]: ax.set_xlabel('MSCE difference vs fixed PPI (x 1,000); lower is better')
fig.suptitle('Same candidate bank; changes in selection do not reliably improve coverage',fontweight='bold',fontsize=14,y=1.015)
fig.text(.5,-.025,'15 paired seeds per dataset. Points: mean difference. Bars: unadjusted 95% paired bootstrap intervals; split variation on fixed data.',ha='center',fontsize=9)
fig.tight_layout(); save(fig,'selection_msce')

h=pd.read_csv(A/'helpsteer_paired.csv')
fig,axes=plt.subplots(1,2,figsize=(11.6,4.6))
for j,method in enumerate(['Flash-PPI','Pro-PPI']):
    for k,metric in enumerate(['true_pinball','cross_calibrated_pinball_test']):
        row=h[(h.method==method)&(h.metric==metric)].iloc[0]
        axes[0].errorbar(row.difference,j+(.13 if k==0 else -.13),xerr=[[row.difference-row.ci95_low],[row.ci95_high-row.difference]],fmt='o' if k==0 else 's',color=COLORS[k],capsize=4,label=['Raw risk','Cross-calibrated risk'][k] if j==0 else None)
    row=h[(h.method==method)&(h.metric=='msce')].iloc[0]
    axes[1].errorbar(row.difference,j,xerr=[[row.difference-row.ci95_low],[row.ci95_high-row.difference]],fmt='o',color=COLORS[0],capsize=4)
for ax in axes:
    ax.axvline(0,color='#6f7780',ls='--',lw=1); ax.set_yticks([0,1],['Flash-PPI','Pro-PPI']); ax.set_ylim(-.6,1.6); ax.grid(axis='x',alpha=.15)
axes[0].set_title('Pinball risk difference vs RCP',loc='left'); axes[0].set_xlabel('Lower is better (score units)'); axes[0].legend(frameon=False,loc='lower right')
axes[1].set_title('Final unweighted K10 MSCE difference',loc='left'); axes[1].set_xlabel('Lower is better (squared coverage units)')
fig.suptitle('HelpSteer2: risk remains higher after temporary calibration, while MSCE is lower',fontsize=13,fontweight='bold',y=1.03)
fig.text(.5,-.03,'20 paired seeds; fixed official test data. Unadjusted 95% split-bootstrap intervals. Prompt dependence limits inferential guarantees.',ha='center',fontsize=9)
fig.tight_layout(); save(fig,'helpsteer_risk')

st=pd.read_csv(A/'statistical.csv')
fig,axes=plt.subplots(1,2,figsize=(11.6,4.8))
cases=['nonzero_positive','nonzero_independent','nonzero_negative','zero_positive']
case_labels=['Nonzero D, strong positive','Nonzero D, independent','Nonzero D, negative','Zero D, strong positive']
for i,case in enumerate(cases):
    sub=st[st.case==case].pivot(index='n',columns='beta',values='variance_total')
    axes[0].plot(sub.index,sub[1.]/sub[0.],marker=['o','s','^','D'][i],color=COLORS[i],label=case_labels[i])
axes[0].axhline(1,color='#6f7780',lw=1,ls='--'); axes[0].set_xscale('log'); axes[0].set_xticks([20,50,100,200],['20','50','100','200']); axes[0].set_xlabel('Trusted validation n (N = 5n)')
axes[0].set_ylabel('Exact variance: beta=1 / beta=0'); axes[0].set_title('Synthetic correction can reduce or increase variance',loc='left',fontsize=11); axes[0].legend(frameon=False,fontsize=8.6)
for beta,color,marker in [(0.,COLORS[0],'o'),(1.,COLORS[1],'s')]:
    sub=st[(st.case=='zero_positive')&(st.beta==beta)].sort_values('n')
    axes[1].loglog(sub.n,sub.variance_total,color=color,label=f'beta={int(beta)} exact')
    axes[1].loglog(sub.n,sub.empirical_variance,ls='none',marker=marker,color=color,ms=6,label=f'beta={int(beta)} Monte Carlo')
axes[1].set_xticks([20,50,100,200],['20','50','100','200']); axes[1].set_xlabel('Trusted validation n (N = 5n)'); axes[1].set_ylabel('Variance (log scale)'); axes[1].set_title('At zero discrepancy, the first-order term vanishes',loc='left',fontsize=11); axes[1].legend(frameon=False,fontsize=8.6)
for ax in axes: ax.grid(alpha=.16)
fig.text(.5,-.025,'Exact finite-support calculation and 100,000 Monte Carlo repetitions per setting. Negative discrepancy estimates are retained.',ha='center',fontsize=9)
fig.tight_layout(); save(fig,'estimator_variance')

head=pd.read_csv(A/'headroom.csv').query('metric=="group_raw_k30"')
fig,ax=plt.subplots(figsize=(10.2,4.8))
vals=[head[head.dataset==ds].headroom.to_numpy()*1000 for ds in DATASETS]
ax.boxplot(vals,positions=np.arange(4),widths=.4,showfliers=False,patch_artist=True,boxprops={'facecolor':'#e3ebf1','edgecolor':'#286b9d'},medianprops={'color':'#27313b'})
rng=np.random.default_rng(33)
for i,v in enumerate(vals): ax.scatter(i+rng.uniform(-.1,.1,len(v)),v,color=COLORS[0],s=25,alpha=.75)
ax.set_xticks(range(4),[NAMES[x] for x in DATASETS]); ax.set_ylabel('Fixed PPI MSCE minus best bank MSCE (x 1,000)'); ax.set_ylim(bottom=-.12); ax.grid(axis='y',alpha=.15)
ax.set_title('Evaluation-only selection headroom is greatest on SGEMM',loc='left',fontsize=13,fontweight='bold')
fig.text(.5,-.025,'15 seeds each. Best of six candidates is chosen retrospectively on evaluation data: an optimistic diagnostic, never a deployed selector.',ha='center',fontsize=9)
fig.tight_layout(); save(fig,'selection_headroom')
print('Saved four scientific figures as SVG and PNG.')
