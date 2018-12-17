import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from misc import penalty


blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'
gray = '#7f7f7f'
brown = '#AA4499'
black = 'black'
scatter_color = red


def plot_network_convergence(ax, network_err, alphas):
    for i in range(len(network_err)):
        ax.plot(network_err[i], label=r'$\alpha={}$'.format(alphas[i]), linewidth=1)

    ax.set_title(r'$\Vert x(B_k) - x_k \Vert^2$')
    ax.set_xlabel(r'$k$')
    ax.set_yscale('log')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_data(ax, pts, y_true, y_delta):
    ax.plot(pts, y_delta, label=r'$y^\delta$', linewidth=1, color=orange)
    ax.plot(pts, y_true, label=r'$y^\dagger$', linestyle='--', linewidth=1, color=blue)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_true_solution(ax, pts, x_true, discrete_plot=False):
    if discrete_plot:
        for i in range(len(pts)):
            ax.scatter(pts[i], x_true[i], marker='o', s=6, color=black)
            ax.plot([pts[i], pts[i]], [0, x_true[i]], linestyle=':', linewidth=1, color=black)
        ax.scatter([], [], label=r'$x^\dagger$', marker='o', s=6, color=black)
    else:
        ax.plot(pts, x_true, label=r'$x^\dagger$', linestyle=':', linewidth=1, color=black)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def plot_lcurve(ax, res_seq, pen_seq, index=None):
    ax.set_title('L-curve')
    ax.plot(res_seq, pen_seq, color=red, label='l-curve', linewidth=1)
    if index is not None:
        ax.scatter(res_seq[index], pen_seq[index], s=7, color=scatter_color, zorder=100)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([], minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_ylabel(r'$\Vert x(B) \Vert^2_2$', labelpad=-2)
    ax.set_xlabel(r'$\Vert Ax(B) - y^\delta \Vert^2_2$', labelpad=-2)


def plot_true_error(ax, true_err1, true_err2, ref_err=None, index=-1):
    ax.set_title('True error')
    ax.plot(true_err1, color=blue, linewidth=1)

    # Uncomment to see if the output of the network is converging
    # if true_err2 is not None:
    #     ax.plot(true_err2, color='black', linewidth=1, linestyle='-.')

    if ref_err is not None:
        ax.axhline(ref_err, 0, len(true_err1), color=orange, linewidth=1, linestyle='--')

    if index != -1 and index is not None:
        ax.scatter(index, true_err1[index], s=7, color=scatter_color, zorder=100)
        ax.set_xticks([0, index, len(true_err1)])
        ax.set_xticklabels([0, '    $k^*$', len(true_err1)])
    else:
        ax.set_xticks([0, len(true_err1)])
        ax.set_xticklabels([0, len(true_err1)])
    ax.set_yscale('log')


def plot_matrix(ax, matrix, label='', make_colorbar=False):
    im = ax.imshow(matrix, cmap='jet')
    ax.set_title(label)
    ax.set_axis_off()
    if make_colorbar:
        plt.colorbar(im, ax=ax, use_gridspec=True)


def plot_alpha(ax, alpha_seq, alpha0, index):
    ax.set_title(r'$\alpha$')
    ax.plot(alpha_seq, linewidth=1)
    if index != -1 and index is not None:
        ax.scatter(index, alpha_seq[index], s=7, color=scatter_color, zorder=100)
    ax.set_yscale('log')

    ax.set_yticks([], minor=True)
    ax.set_yticklabels([], minor=True)
    ax.set_yticks([alpha0, alpha_seq[index]])
    ax.set_yticklabels([r'$\alpha_0$', r'$\alpha_{k*}$'])

    ax.set_xticks([0, index, len(alpha_seq)])
    ax.set_xticklabels([0, '    $k^*$', len(alpha_seq)])


def plot_matrix_convergence(ax, A, B_seq, index=None):
    ax.set_title(r'$\Vert B_{k}-B_{k-1} \Vert_F$')
    seq = [np.linalg.norm(B_seq[i] - B_seq[i-1], 'fro') for i in range(1, len(B_seq))]
    ax.plot(range(1, len(B_seq)), seq, linewidth=1, color=red)
    if index != -1 and index is not None:
        ax.scatter(index, seq[index], s=7, color=scatter_color, zorder=100)
        ax.set_xticks([0, index, len(B_seq)])
        ax.set_xticklabels([0, '    $k^*$', len(B_seq)])
    else:
        ax.set_xticks([0, len(B_seq)])
        ax.set_xticklabels([0, len(B_seq)])
    ax.set_yscale('log')


def plot_reconstructions(ax, pts, x_true, x_ref, x_deep, discrete_plot=False):
    if discrete_plot:
        for i in range(len(pts)):
            if np.abs(x_true[i]) > 1e-9:
                ax.scatter(pts[i], x_true[i], marker='o', s=6, color=black)
                ax.plot([pts[i], pts[i]], [0, x_true[i]], linestyle=':', linewidth=1, color=black)
            ax.scatter(pts[i], x_ref[i], marker='o', s=6, color=orange, facecolors='none', zorder=100)
            ax.scatter(pts[i], x_deep[i], marker='s', s=6, color=blue, facecolors='none', zorder=101)
        ax.scatter([], [], label='$x^\dagger$', marker='o', s=6, color=black)
        ax.scatter([], [], label='$x_T$', marker='o', s=6, color=orange, facecolors='none')
        ax.scatter([], [], label='$x(B_{opt})$', marker='s', s=6, color=blue, facecolors='none')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.plot(pts, x_true, label='$x^\dagger$', linestyle=':', linewidth=1, color=black)
        ax.plot(pts, x_ref, color=orange, label='$x_T$', linestyle='--', linewidth=1, alpha=0.7)
        ax.plot(pts, x_deep, label='$x(B_{opt})$', linewidth=1, color=blue, alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def _plot_summary_fixed(pts, A, x_true, x_ref, B_seq, x_seq, alpha,  true_err1, true_err2, axes, index=-1, discrete_plot=False, make_colorbar=False):
    for i in range(len(axes)):
        axes[i].clear()
    if index != -1:
        axes[0].set_title(r'Reconstructions ($\alpha={}, k^*={}$)'.format(alpha, index))
    else:
        axes[0].set_title(r'Reconstructions ($\alpha={}$)'.format(alpha))

    plot_reconstructions(axes[0], pts, x_true, x_ref, x_seq[index], discrete_plot=discrete_plot)
    plot_true_error(axes[1], true_err1, true_err2, ref_err=penalty(x_true - x_ref, 'l2'), index=index)
    plot_matrix_convergence(axes[2], A, B_seq, index=index)
    plot_matrix(axes[3], B_seq[index], label=r'$B_{opt}$', make_colorbar=make_colorbar)


def plot_summary_fixed(pts, A, alpha, x_true, x_ref, B_seq, x_seq, x_seq_tik=None, discrete_plot=False):
    fig = plt.figure(figsize=(11.5, 2.2))
    ax = []
    gs1 = gridspec.GridSpec(1, 1)
    ax += [fig.add_subplot(gs1[0])]
    gs2 = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax += [fig.add_subplot(gs2[0])]
    ax += [fig.add_subplot(gs2[1])]
    gs3 = gridspec.GridSpec(1, 1)
    ax += [fig.add_subplot(gs3[0])]
    true_err1 = list([penalty(x_true - x_seq[i]) for i in range(len(x_seq))])

    if x_seq_tik is not None:
        true_err2 = list([penalty(x_true - x_seq_tik[i]) for i in range(len(x_seq_tik))])
    else:
        true_err2 = None

    _plot_summary_fixed(pts, A, x_true, x_ref, B_seq, x_seq, alpha=alpha, true_err1=true_err1, true_err2=true_err2, axes=ax, discrete_plot=discrete_plot, make_colorbar=True)

    gs1.tight_layout(fig, rect=[0, 0, 0.46, 0.99])
    gs2.tight_layout(fig, rect=[0.45, 0, 0.79, 1], w_pad=1.0)
    gs3.tight_layout(fig, rect=[0.65, 0, 1.0, 1.00])

    def onclick(event):
        current = event.inaxes
        if current is ax[2]:
            index = int(event.xdata)
            _plot_summary_fixed(pts, A, x_true, x_ref, B_seq, x_seq, alpha=alpha, true_err1=true_err1, true_err2=true_err2, axes=ax, index=index, discrete_plot=discrete_plot)
            fig.canvas.draw()
            print(index)
    fig.canvas.mpl_connect('button_press_event', onclick)


def _plot_summary_adaptive(pts, A, x_true, x_ref_seq, x_seq, B_seq, alpha0, alpha_seq, true_err1, true_err2, axes, index=-1, make_colorbar=False, discrete_plot=False):
    for i in range(len(axes)):
        axes[i].clear()
    if index != -1:
        axes[0].set_title(r'Reconstructions ($\alpha_0={},k^*={}, \alpha_{}={:.2e}$)'.format(alpha0, index, '{k*}', alpha_seq[index]))
    else:
        axes[0].set_title(r'Reconstructions ($\alpha_0={}$)'.format(alpha0))

    plot_reconstructions(axes[0], pts, x_true, x_ref_seq[index], x_seq[index], discrete_plot=discrete_plot)
    plot_true_error(axes[1], true_err1, true_err2, ref_err=penalty(x_true - x_ref_seq[index], 'l2'), index=index)
    plot_alpha(axes[2], alpha_seq, alpha0, index)
    plot_matrix(axes[3], B_seq[index], label=r'$B_{k*}$', make_colorbar=make_colorbar)


def plot_summary_adaptive(pts, A, alpha0, alpha_seq, x_true, B_seq, x_seq, ref_seq, x_seq_tik=None, index=-1, discrete_plot=False):
    fig = plt.figure(figsize=(11.5, 2.2))
    ax = []
    gs1 = gridspec.GridSpec(1, 1)
    ax += [fig.add_subplot(gs1[0])]
    gs2 = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax += [fig.add_subplot(gs2[0])]
    ax += [fig.add_subplot(gs2[1])]
    gs3 = gridspec.GridSpec(1, 1)
    ax += [fig.add_subplot(gs3[0])]

    true_err1 = list([penalty(x_true - x_seq[i]) for i in range(len(x_seq))])
    if x_seq_tik is not None:
        true_err2 = [penalty(x_true - x_seq_tik[i]) for i in range(len(B_seq))]
    else:
        true_err2 = None

    _plot_summary_adaptive(pts, A, x_true, x_ref_seq=ref_seq, x_seq=x_seq, B_seq=B_seq, alpha0=alpha0, alpha_seq=alpha_seq, true_err1=true_err1, true_err2=true_err2, index=index, axes=ax, make_colorbar=True, discrete_plot=discrete_plot)

    gs1.tight_layout(fig, rect=[0, 0, 0.46, 0.99])
    gs2.tight_layout(fig, rect=[0.45, 0, 0.79, 1], w_pad=1.0)
    gs3.tight_layout(fig, rect=[0.65, 0.01, 1.0, 1.00])

    def onclick(event):
        current = event.inaxes
        if current is ax[2] or current is ax[3]:
            index = int(event.xdata)

            _plot_summary_adaptive(pts, A, x_true, x_ref_seq=ref_seq, x_seq=x_seq, B_seq=B_seq, 
                                   alpha0=alpha0, alpha_seq=alpha_seq, true_err1=true_err1, 
                                   true_err2=true_err2, axes=ax, index=index, 
                                   discrete_plot=discrete_plot)

            print('selected iteration (k*): {}'.format(index), end='\r')
            print('standard tikhonov error: {}'.format(penalty(ref_seq[index] - x_true)), end='\r')
            print('analytic deep inverse prior error: {}'.format(penalty(x_seq[index] - x_true)), end='\r')

            fig.canvas.draw()
            print(index)
    fig.canvas.mpl_connect('button_press_event', onclick)
