import argparse
import numpy as np
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

from config_selection import logger
from config_selection.score_parser import ScoreParser


def _write_scores_to_plot(scores, destination, **kwargs):
    def _add_plot(score, title):
        to_plot = np.concatenate([[0], np.mean(score.pertubation_scores, axis=0)], axis=0)
        plt.plot(np.arange(score.pertubations + 1), to_plot, label=title, marker=kwargs['marker'])

    if kwargs['line_titles']:
        for (score, title) in zip(scores, kwargs['line_titles']):
            _add_plot(score, title)
    else:
        for score in scores:
            _add_plot(score, score.short_description())

    plt.ylabel('Score differences')
    plt.xlabel('Pertubations')
    if len(kwargs['plot_title']) > 0:
        plt.title(kwargs['plot_title'])

    plt.legend(loc=2)

    plt.savefig(destination, bbox_inches='tight')
    logger.debug("Graph generated at {}".format(destination))


def _show_results(**kwargs):
    if kwargs['file']:
        scores = [ScoreParser(f) for f in kwargs['file']]
    else:
        scores = [ScoreParser(join(kwargs['benchmark_dir'], f)) for f in listdir(kwargs['benchmark_dir']) if
                  isfile(join(kwargs['benchmark_dir'], f)) and f.endswith(".res")]

    scores.sort(key=lambda x: x.AOPC, reverse=True)

    for score in scores:
        logger.info("--------------------------------------------------")
        logger.info(score)

    if kwargs['plot']:
        _write_scores_to_plot(scores, **kwargs)


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
