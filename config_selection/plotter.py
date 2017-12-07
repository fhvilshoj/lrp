import numpy as np

from config_selection import logger

from matplotlib import rcParams
rcParams['font.family'] = 'serif'

from matplotlib import pyplot as plt
from matplotlib import patches

def write_scores_to_plot(scores, destination, **kwargs):
    label_font = {
        'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
    }

    title_font = {
        'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
    }

    ticks_font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 12,
    }
    
    def _add_plot(score, title):
        to_plot = np.concatenate([score.x0_predictions, score.pertubation_scores], axis=0)
        plt.plot(np.arange(score.pertubations + 1), to_plot, label=title, marker=kwargs['marker'])
        
    plt.axhline(y=scores[0].x0_predictions[0], xmin=0, xmax=100, linewidth=0.8, color = 'tab:gray', linestyle='--')

    if kwargs['line_titles']:
        for (score, title) in zip(scores, kwargs['line_titles']):
            _add_plot(score, title)
    elif kwargs['best']:
        scores = scores[:3]
        scores = [scores[i] for i in [2, 0, 1]]
        for score in scores:
            desc = score.short_description()
            desc = 'LRP' if 'LIN_' in desc else desc
            _add_plot(score, desc)
    else:
        for score in scores:
            _add_plot(score, score.short_description())

    plt.ylabel('Prediction score', fontdict=label_font)
    plt.xlabel('Pertubations', fontdict=label_font)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    
    if len(kwargs['plot_title']) > 0:
        plt.text(45, 1.05, kwargs['plot_title'], fontdict=title_font)
        #plt.title(kwargs['plot_title'], fontdict=title_font, bbox_to_anchor=(0., 1.1, 1., .102))

    leg = plt.legend(bbox_to_anchor=(0., -.25, 1., .102), loc=2,
                     ncol=3, mode="expand", borderaxespad=0., prop={'size': 13}) # (loc=3)
    frame = leg.get_frame()
    frame.set_linewidth(0.0)
    frame.set_facecolor((1., 1., 1., 0.))
    frame.set_alpha(0.)    
    
    aopc_y_pos = -.26
    aopc_x_pos = [11.8, 47.7, 87.7]

    plt.text(-10, aopc_y_pos, r'$AOPC:$')

    if kwargs['best']:
        scores = scores[:3]    
        for pos, score in zip(aopc_x_pos, scores):
            plt.text(pos, aopc_y_pos, "{:.2f}".format(score.AOPC))
    
    plt.savefig(destination, bbox_inches='tight')
    logger.debug("Graph generated at {}".format(destination))
