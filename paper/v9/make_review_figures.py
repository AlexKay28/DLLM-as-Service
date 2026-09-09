"""Build the inherited v8 figures in this snapshot from committed results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
RESULTS=REPO/'results'
def read(path): return json.loads((RESULTS/path).read_text())
def latencies(path):
    return np.array([json.loads(line)['latency_ms'] for line in (RESULTS/path).read_text().splitlines() if line.strip()])

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    arcolor='#2A78D6'; dgcolor='#D75B2F'
    stats=read('anchors/ar30b_stats.json')
    fit=read('issue111_qwen3_serial_sweep.json')['cost_law']
    base=fit['dg_base_ms']; step=fit['dg_step_ms']; a=fit['intercept_ms']; b=fit['slope_ms_per_token']
    x=np.arange(1,2049)
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
    ax[0].plot(x[x<=126],(a+b*x[x<=126])/1000,color=arcolor,label='AR fit: measured length range')
    ax[0].plot(x[x>=126],(a+b*x[x>=126])/1000,color=arcolor,ls='--',label='AR extrapolation')
    ax[0].plot(x,(base+step*(np.ceil(x/256)-1))/1000,color=dgcolor,label='DG canvas-cost approximation')
    # Actual canvas measurements are retained as points.
    xs=[256,512,768,1024,1280,1536,1792,2048]
    ys=[]
    for length in xs:
        d=read(f'issue68_dg_L{length}.json')
        ys.append(next(iter(d['models'].values()))['streaming_table3']['latency']['p50_ms']/1000)
    ax[0].scatter(xs,ys,color=dgcolor,s=20,zorder=3)
    ax[0].set(xlabel='Output length / full-canvas budget (tokens)',ylabel='Service cost (s, log scale)',yscale='log',title='A  Cost-model range and extrapolation',xlim=(0,2080))
    ax[0].legend(fontsize=7.5,loc='upper left')
    ax[0].grid(alpha=.15)
    ds=[stats['ar30b_bgl'],stats['dg_bgl']]
    med=np.array([d['p50_ms'] for d in ds]); lo=np.array([d['p50_ci_95'][0] for d in ds]); hi=np.array([d['p50_ci_95'][1] for d in ds])
    ax[1].bar(['AR-30B','DG'],med,color=[arcolor,dgcolor],width=.55)
    ax[1].errorbar([0,1],med,yerr=[med-lo,hi-med],fmt='none',color='black',capsize=5)
    for i,d in enumerate(ds):ax[1].text(i,med[i]+40,f"{med[i]:.1f} ms\nn = {d['n_events']:,}",ha='center',fontsize=9)
    ax[1].set(ylabel='Median latency (ms)',ylim=(0,1130),title='B  Observed serial medians')
    fig.savefig(HERE/'figures/fig1_v8.pdf'); plt.close(fig)
    ar=np.concatenate([latencies(f'anchors/ar30b_L256_s{s}_preds.jsonl') for s in [42,137,314]])
    dg=latencies('anchors/dg_L256_r1_preds.jsonl')
    fig,ax=plt.subplots(figsize=(8,3.7),layout='constrained')
    deadline=1000
    for data,label,color,key in [(ar,'AR-30B (900 recorded events)',arcolor,'ar30b_bgl'),(dg,'DG (2,000 recorded events)',dgcolor,'dg_bgl')]:
        vals=np.sort(data)
        ax.step(vals,np.arange(1,len(vals)+1)/len(vals)*100,where='post',color=color,label=label,lw=1.8)
        completed=float(np.mean(vals<=deadline)*100)
        ax.hlines(completed,100,deadline,color=color,ls=':',lw=.8,alpha=.7)
        ax.scatter([deadline],[completed],color=color,s=35,zorder=4)
        ax.annotate(f'{completed:.2f}% by 1,000 ms',(deadline,completed),
                    xytext=(1120,88 if key=='dg_bgl' else 64),
                    arrowprops={'arrowstyle':'->','color':color},color=color,fontsize=10)
    ax.axvline(deadline,color='gray',lw=.8,ls='--')
    ax.set(xlabel='Generation-time budget (ms)',ylabel='Generations completed (%)',
           ylim=(0,104),xlim=(100,1700),yticks=[0,25,50,75,100])
    ax.legend(loc='lower right');ax.grid(alpha=.15)
    fig.savefig(HERE/'figures/fig12_v8.pdf');plt.close(fig)
    print('Built fig1_v8.pdf and fig12_v8.pdf from committed artifacts.')

if __name__=='__main__':main()
