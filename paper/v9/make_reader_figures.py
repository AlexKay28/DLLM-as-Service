"""Build reader diagrams and the chain ratio plot; no new inference."""
from make_review_figures import HERE, read
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

AR = '#2A78D6'
DG = '#D75B2F'


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015',
                              facecolor=color, edgecolor='white', alpha=.14))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=11)


def save(fig, name):
    fig.savefig(HERE/'figures'/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def chain_ratio():
    data = read('chain_analysis.json')
    tail = data['P_E2_tail_concentration']
    linear = data['P_E1_linearity']
    depths = tail['depths']
    assert depths == linear['ar']['depths'] == linear['dg']['depths']
    ar = [t*d/1000 for t, d in zip(linear['ar']['cumul_p50_per_depth'], depths)]
    dg = [t*d/1000 for t, d in zip(linear['dg']['cumul_p50_per_depth'], depths)]
    ratios = [a/d for a, d in zip(ar, dg)]
    assert all(abs(a-b) < 1e-9 for a, b in zip(ratios, tail['p50_ratio_per_depth']))
    fig, ax = plt.subplots(figsize=(9.4, 3.5), layout='constrained')
    # Evenly spaced measured depths leave room for absolute-time labels.
    x = list(range(len(depths)))
    ax.plot(x, ratios, 'o-', color=AR, lw=2, label='Median ratio')
    ax.plot(x, tail['p999_ratio_per_depth'], 's--', color=DG,
            label='p99.9 ratio (exploratory)')
    ax.axhline(1, color='gray', lw=.8)
    for i, (a, d, r) in enumerate(zip(ar, dg, ratios)):
        ax.annotate(f'{r:.2f}×\n{a:.2f} / {d:.2f} s', (i, r),
                    xytext=(0, 18), textcoords='offset points', ha='center', fontsize=10)
    ax.set(xticks=x, xticklabels=depths, xlabel='Chain depth (dependent calls)',
           ylabel='AR-3B / DG cumulative latency', ylim=(.85, 2.62), xlim=(-.35, 4.35),
           title='Median chain latency stays near 2× lower for diffusion')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=.15)
    save(fig, 'fig10_usecase_chains')


def decoding():
    fig, ax = plt.subplots(figsize=(9.4, 2.4))
    ax.set(xlim=(0, 10), ylim=(0, 3)); ax.axis('off')
    ax.text(0, 2.65, 'Autoregressive: append one token per step', weight='bold', fontsize=12)
    for i in range(5):
        box(ax, .15+i*1.4, 1.85, 1.05, .55, f'Token {i+1}', AR)
        if i < 4:
            ax.annotate('', (.15+(i+1)*1.4, 2.12), (1.2+i*1.4, 2.12),
                        arrowprops={'arrowstyle':'->', 'color':AR})
    ax.text(7.5, 2.1, 'Sequential\ntoken steps', va='center', fontsize=11, color=AR)
    ax.text(0, 1.35, 'Block diffusion: refine positions together, then advance', weight='bold', fontsize=12)
    box(ax, .15, .35, 3.35, .65, 'Canvas 1: K refinement passes', DG)
    box(ax, 4.05, .35, 3.35, .65, 'Canvas 2: K refinement passes', DG)
    ax.annotate('', (4.0, .68), (3.55, .68), arrowprops={'arrowstyle':'->', 'color':DG})
    ax.text(7.7, .68, 'Sequential\ncanvases', va='center', fontsize=11, color=DG)
    save(fig, 'decoding_schematic')


def deadline():
    fig, ax = plt.subplots(figsize=(9.4, 2.3))
    ax.set(xlim=(0, 10), ylim=(0, 2.7)); ax.axis('off')
    for i, (label, color) in enumerate([('Queue', '#64748B'), ('Generate', DG),
                                       ('Validate record', '#16866A'), ('Act / decide', '#64748B')]):
        x = .2+i*2.42
        box(ax, x, 1.15, 2.1, .65, label, color)
        if i < 3:
            ax.annotate('', (x+2.42, 1.48), (x+2.1, 1.48), arrowprops={'arrowstyle':'->'})
    ax.annotate('', (9.58, 2.18), (.2, 2.18), arrowprops={'arrowstyle':'<->'})
    ax.text(4.9, 2.38, 'One response budget: from event arrival to decision', ha='center', fontsize=12)
    ax.text(3.65, .86, 'Measured generation time', ha='center', color=DG, fontsize=10)
    ax.text(4.9, .3, 'Finished + valid + acceptable quality + on time → usable response',
            ha='center', fontsize=12, weight='bold')
    save(fig, 'deadline_path')


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    chain_ratio()
    decoding()
    deadline()
    print('Built chain ratios from saved results and two conceptual diagrams.')


if __name__ == '__main__':
    main()
