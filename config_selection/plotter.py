import numpy as np

from config_selection import logger

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True

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

    leg = plt.legend(bbox_to_anchor=(0., -.25, 1., .102), loc=2, labelspacing=1.2,
                     ncol=kwargs['legend_columns'], mode="expand", borderaxespad=0., prop={'size': 13}) # (loc=3)
    frame = leg.get_frame()
    frame.set_linewidth(0.0)
    frame.set_facecolor((1., 1., 1., 0.))
    frame.set_alpha(0.)    
    
    aopc_y_pos = -.245

    cols = kwargs['legend_columns']

    col_height = len(scores) // cols
    spare = len(scores) % cols
    col_height = col_height if spare == 0 else col_height + 1

    for i in range(col_height):
        plt.text(-10, aopc_y_pos + i * -0.1, r'$AOPC:$')

    columns = []
    spare_used = 0
    score_idx = 0
    for i in range(cols):
        column = []
        if spare_used == spare:
            col_height -= 1
        for j in range(col_height):
            column.append(scores[score_idx])
            score_idx += 1
        columns.append(column)
        spare_used += 1

    print("Cols: {cols} Col height: {col_height} Scores {scores}".format(cols=cols, col_height=col_height, scores=len(scores)))

    if kwargs['best']:
        aopc_x_pos = [11.8, 47.7, 87.7]
        scores = scores[:3]
        for pos, score in zip(aopc_x_pos, scores):
            plt.text(pos, aopc_y_pos, "{:.2f}".format(score.AOPC))
    else:
        if cols == 2:
            x_start = 18
            x_offset = 62
        else:
            x_start = 14.5
            x_offset = 37.9

        for c_off, col in enumerate(columns):
            x_pos = x_start + c_off * x_offset
            for sc_off, sc in enumerate(col):
                y_pos = aopc_y_pos + sc_off * -0.1
                plt.text(x_pos, y_pos, "{:.2f}".format(sc.AOPC))

        # for idx, score in enumerate(scores):
        #     x_pos = x_start + x_offset * (idx // col_height)
        #     y_pos = aopc_y_pos + (idx % col_height) * -0.1
        #     plt.text(x_pos, y_pos, "{:.2f}".format(score.AOPC))

    plt.savefig(destination, bbox_inches='tight')
    logger.debug("Graph generated at {}".format(destination))
