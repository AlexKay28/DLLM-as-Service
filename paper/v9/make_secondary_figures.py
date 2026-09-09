"""Render secondary figures in this snapshot from committed data."""
from make_review_figures import HERE, read
import matplotlib.pyplot as plt
import numpy as np

AR='#2A78D6'; DG='#D75B2F'; LL='#16866A'
def cell(name): return next(iter(read(name)['models'].values()))['streaming_table3']
def p50(name): return cell(name)['latency']['p50_ms']
def save(fig,name):
    for ax in fig.axes: ax.grid(alpha=.15); ax.set_axisbelow(True)
    fig.savefig(HERE/'figures'/f'{name}.pdf');plt.close(fig)
def panels(): return plt.subplots(1,2,figsize=(9.4,3.3),layout='constrained')

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=panels(); lens=[240,256,264,496,512,528]
    ax[0].bar(range(len(lens)),[p50(f'issue68_dg_L{x}.json') for x in lens],color=DG)
    ax[0].set(xticks=range(len(lens)),xticklabels=lens,xlabel='Forced output budget (tokens)',ylabel='Median latency (ms)',title='A  Boundary probes')
    full=list(range(256,2049,256)); vals=[p50(f'issue68_dg_L{x}.json') for x in full]
    ax[1].bar(range(7),np.diff(vals),color=DG)
    ax[1].set(xticks=range(7),xticklabels=[f'{i}→{i+1}' for i in range(1,8)],xlabel='Canvas-count transition',ylabel='Increment in median latency (ms)',title='B  Measured canvas increments',ylim=(0,150))
    save(fig,'fig2_canvas_edges')

    fig,ax=panels(); slopes=[]
    for arm,color,label in [('ar3b',AR,'AR-3B (dense)'),('dg',DG,'DG')]:
        sweep=read(f'issue44_{arm}_input_sweep.json')['sweep'];x=[d['prompt_token_count'] for d in sweep];y=[d['latency_ms']['p50'] for d in sweep]
        slopes.append(np.polyfit(x,y,1)[0]);ax[0].plot(x,y,'o-' if arm=='ar3b' else 's--',color=color,label=label)
    ax[0].set(xlabel='Prompt tokens',ylabel='Median latency (ms)',title='A  Input-length dependence');ax[0].legend()
    ax[1].barh(['AR-3B (dense)','DG'],slopes,color=[AR,DG]);ax[1].set(xlabel='Fitted latency slope (ms/prompt token)',title='B  Empirical input slopes',xlim=(0,.15))
    for i,s in enumerate(slopes): ax[1].text(s+.003,i,f'{s:.3f}',va='center')
    save(fig,'fig3_input_sweep')

    fig,ax=panels();ks=[2,4,8,16,32,64]; cs=[cell(f'issue44_dg_ksweep_K{k}.json') for k in ks]
    for axis,vals,title,ylabel in [(ax[0],[c['latency']['p50_ms'] for c in cs],'A  Latency versus configured effort','Median latency (ms)'),(ax[1],[c['quality'].get('field_f1_9',c['quality']['field_f1']) for c in cs],'B  Quality versus configured effort','9-field F1')]:
        axis.plot(ks,vals,'s-',color=DG);axis.set(xscale='log',xlabel='Configured denoising steps K',ylabel=ylabel,title=title,xticks=ks,xticklabels=ks);axis.minorticks_off();axis.axvline(8,color='gray',ls=':',lw=.8)
    save(fig,'fig4_ksweep')

    fig,ax=panels();lengths=[32,128,256,512];nine=read('issue43_nine_field_f1.json')
    for arm,color,label in [('ar3b',AR,'AR-3B (dense)'),('dg',DG,'DG')]:
        f=[nine[f'{arm}_L{x}']['field_f1_9'] for x in lengths];v=[cell(f'issue43/{arm}_L{x}.json')['quality']['schema_valid_pct'] for x in lengths]
        for axis,ys in zip(ax,[f,v]):axis.plot(range(4),ys,'o-' if arm=='ar3b' else 's--',color=color,label=label);axis.set(xticks=range(4),xticklabels=lengths,xlabel='Output budget (tokens)')
    ax[0].set(ylabel='9-field F1',title='A  Extraction quality',ylim=(0,1));ax[1].set(ylabel='Schema-valid records (%)',title='B  Harness validity',ylim=(0,105));ax[0].legend()
    save(fig,'fig5_quality_vs_length')

    fig,ax=plt.subplots(1,2,figsize=(10.5,4),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
    cfg=read('issue42_batching.json')['configs'];lla=max(x['multiplier'] for x in read('llada_dinfer/batch_sweep.json')['results'])
    labels=['AR-30B, vLLM','AR-3B, prefix cache on','AR-3B, prefix cache off','DG, tested vLLM (B=128)','LLaDA-8B, cache-less dInfer']
    vals=[read('issue111_qwen3_batch_sweep.json')['multiplier'],cfg['caching_on']['multiplier'],cfg['caching_off']['multiplier'],1.7,lla]
    ax[0].barh(range(5),vals,color=[AR,AR,AR,DG,LL]);ax[0].invert_yaxis();ax[0].set(yticks=range(5),yticklabels=labels,xlabel='Observed batch-1 to peak multiplier',xscale='log',xlim=(.8,85),title='A  Observed batching gains')
    for i,v in enumerate(vals):ax[0].text(v*1.08,i,f'{v:g}×',va='center',fontsize=8)
    for name,style in [('caching_on','o-'),('caching_off','s--')]:
        rows=cfg[name]['sweep'];ax[1].plot([x['batch_size'] for x in rows],[x['events_per_s'] for x in rows],style,color=AR,label='Prefix cache '+('on' if name.endswith('_on') else 'off'))
    ax[1].set(xscale='log',xlabel='Batch size',ylabel='Throughput (events/s)',title='B  Dense AR-3B cache control');ax[1].legend(fontsize=8)
    save(fig,'fig6_batching')

    fig,ax=panels();data=read('chain_analysis.json');linear=data['P_E1_linearity'];tail=data['P_E2_tail_concentration']
    for name,color,label in [('ar',AR,'AR-3B (dense)'),('dg',DG,'DG')]:ax[0].plot(linear[name]['depths'],linear[name]['cumul_p50_per_depth'],'o-' if name=='ar' else 's--',color=color,label=label)
    ax[0].set(xlabel='Chain depth',ylabel='Cumulative median / depth (ms)',title='A  Normalized chain latency');ax[0].legend()
    ax[1].plot(tail['depths'],tail['p999_ratio_per_depth'],'s-',color=DG);ax[1].set(xlabel='Chain depth',ylabel='AR/DG cumulative p99.9 ratio',title='B  Exploratory tail ratio',ylim=(.8,2.3))
    save(fig,'fig10_usecase_chains')

    fig,ax=plt.subplots(figsize=(7.8,3.4),layout='constrained');data=read('wide_schema_analysis.json')['P_C1_latency']
    for name,color,label in [('ar',AR,'AR-3B (dense)'),('dg',DG,'DG')]:ax.plot([d['n_fields'] for d in data[name]],[d['p50_ms'] for d in data[name]],'o-' if name=='ar' else 's--',color=color,label=label)
    ax.set(xlabel='Schema width (fields)',ylabel='Median latency (ms)',title='Schema-width workload comparison',xticks=[12,25,50,100]);ax.legend();save(fig,'fig11_usecase_schema')
    print('Built seven secondary figures with corrected labels.')

if __name__=='__main__':main()
