import argparse
import numpy as np
from matplotlib import pyplot as plt

from config_selection.plotter import write_scores_to_plot

from os import listdir
from os.path import isfile, join

from config_selection import logger
from config_selection.score_parser import ScoreParser


def _show_results(**kwargs):
    if kwargs['file']:
        scores = [ScoreParser(f) for f in kwargs['file']]
    else:
        scores = [ScoreParser(join(kwargs['benchmark_dir'], f)) for f in listdir(kwargs['benchmark_dir']) if
                  isfile(join(kwargs['benchmark_dir'], f)) and f.endswith(".res")]

    scores.sort(key=lambda x: x.AOPC, reverse=True)
    to_print = [sc for sc in scores if sc.title in ['Sensitivity Analysis', 'Random']]
    rest = [sc for sc in scores if sc.title not in ['Sensitivity Analysis', 'Random']]

    to_print.extend(rest[:max(10, len(rest))])

    for score in to_print:
        logger.info("--------------------------------------------------")
        logger.info("\n" + str(score))

    if kwargs['plot']:
        write_scores_to_plot(to_print, **kwargs)


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
