import argparse
import numpy as np
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

from config_selection import logger
from config_selection.score_parser_mnist import ScoreParser


def _write_scores_to_plot(scores, destination, **kwargs):
    def _add_plot(score, title):
        to_plot = np.concatenate([score.x0_predictions, score.pertubation_scores], axis=0)
        plt.plot(np.arange(score.pertubations + 1), to_plot, label=title, marker=kwargs['marker'])

    plt.axhline(y=scores[0].x0_predictions[0], xmin=0, xmax=100, linewidth=0.8, color = 'tab:gray', linestyle='--')
        
    if kwargs['line_titles']:
        for (score, title) in zip(scores, kwargs['line_titles']):
            _add_plot(score, title)
    elif kwargs['best']:
        for score in scores[:3]:
            desc = score.short_description()
            desc = 'LRP' if 'LIN_' in desc else desc
            _add_plot(score, desc)
    else:
        for score in scores:
            _add_plot(score, score.short_description())

    plt.ylabel('Prediction score')
    plt.xlabel('Pertubations')
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    
    if len(kwargs['plot_title']) > 0:
        plt.title(kwargs['plot_title'])

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.) # (loc=3)

    plt.savefig(destination, bbox_inches='tight')
    logger.debug("Graph generated at {}".format(destination))


def _show_results(**kwargs):
    if kwargs['file']:
        scores = [ScoreParser(f) for f in kwargs['file']]
    else:
        scores = [ScoreParser(join(kwargs['benchmark_dir'], f)) for f in listdir(kwargs['benchmark_dir']) if
                  isfile(join(kwargs['benchmark_dir'], f)) and f.endswith(".res")]

    scores.sort(key=lambda x: x.AOPC, reverse=True)
    to_print = [sc for sc in scores if sc.title in ['Sensitivity Analysis', 'Random']]
    rest = [sc for sc in scores if sc.title not in ['Sensitivity Analysis', 'Random']]

    to_print.extend(rest[:min(10, len(rest))])

    for score in to_print:
        logger.info("--------------------------------------------------")
        logger.info("\n" + str(score))

    if kwargs['plot']:
        _write_scores_to_plot(to_print, **kwargs)


def _main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get model scores')
    parser.add_argument('-b', '--benchmark_dir', type=str,
                        default='E:/frederikogbenjamin/benchmarks/lrp_vs_sa',
                        help='The location of the benchmark results')
    parser.add_argument('-f', '--file', nargs="*", default=[],
                        help='Files to display scores for')
    parser.add_argument('-d', '--destination')

    # Extra options
    parser.add_argument('--plot', action='store_true',
                        help='Boolean indicator. If true plots are generated.')
    parser.add_argument('--best', action='store_true',
                        help='Plot the best lrp rule along with SA and random')
    parser.add_argument('-l', '--line-titles', type=str, nargs="*", default=[],
                        help='Titles to be used in plot')
    parser.add_argument('-m', '--marker', type=str, default='.')
    parser.add_argument('-t', '--plot-title', type=str, default='',
                        help='Title to be displayed in top of plot')

    args = parser.parse_args()

    # Call config selection with gathered arguments
    _show_results(**vars(args))


if __name__ == '__main__':
    _main()
