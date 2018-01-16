import re
import numpy as np
from lrp.configuration import *

# Lookup at linear, batch_norm, sparse, convolution, max_pool, lstm
type_config_re = r'.*/LIN_(?P<linear>.*)_ELE_(?P<batch_norm>.*)_SPA_(?P<sparse>.*)_CONV_(?P<convolution>.*)_MAX_(?P<max_pool>.*)_LSTM_(?P<lstm>.*)'
type_config_sm_re = r'.*/LIN_(?P<linear>.*)_ELE_(?P<batch_norm>.*)_SPA_(?P<sparse>.*)_CONV_(?P<convolution>.*)_MAX_(?P<max_pool>.*)_LSTM_(?P<lstm>.*)_SM_(?P<softmax>.+?)_?(ZB_(?P<zb>.*?))?\.?res'

# Lookup at alpha, epsilon, bias, winners_take_all, flat, ww
rules_re = r'(((a(?P<alpha>\-{0,1}[0-9]+\.*[0-9]*)b(?P<beta>\-{0,1}[0-9]+\.*[0-9]*))|(e(?P<epsilon>[0-9]+\.*[0-9]*))|(?P<zb>l(?P<low>\-?[0-9]+\.?[0-9]*?)_h(?P<high>\-?[0-9]+\.?[0-9]*?)))_(?P<bias>\w\w))|(?P<winners_take_all>wins)|(?P<flat>flat)|(?P<ww>ww)|(?P<winner_takes_all>win)|(?P<identity>id)|(?P<naive>nai)|(?P<none>no)'

# File name
file_name_re = r'.*/(LIN.*\.res)'

bias_strategies = {
    'no': 'Absorb',
    'ig': 'Ignore',
    'ac': 'Active',
    'al': 'All'
}

class Config(object):

    def __init__(self, layer, conf_string):
        try:
            rule_dict = re.match(rules_re, conf_string).groupdict()
        except:
            print("Conf string could not be passed: {} for layer {}".format(conf_string,layer))
            return

        self.layer = layer.replace("_", " ").title()
        self.bias_strategy = None

        if rule_dict['alpha'] is not None:
            self.rule = 'Alpha {alpha} Beta {beta}'.format(**rule_dict)
            self.bias_strategy = bias_strategies[rule_dict['bias']]
        elif rule_dict['epsilon'] is not None:
            self.rule = 'Epsilon {epsilon}'.format(**rule_dict)
            self.bias_strategy = bias_strategies[rule_dict['bias']]
        elif rule_dict['zb'] is not None:
            self.rule = 'Zb'
            self.layer = 'Zb on first'
            self.bias_strategy = bias_strategies[rule_dict['bias']]
        elif rule_dict['flat'] is not None:
            self.rule = 'Flat'
        elif rule_dict['winners_take_all'] is not None or rule_dict['none'] is not None:
            self.rule = 'Winners take all'
        elif rule_dict['identity'] is not None:
            self.rule = 'Identity'
        elif rule_dict['naive'] is not None:
            self.rule = 'Naive'
        else:
            self.rule = 'WW'

    def __str__(self):
        s = "{:12}: {}".format(self.layer, self.rule)
        if self.bias_strategy is not None:
            s += ", bias: {}".format(self.bias_strategy)
        return s


class ScoreParser(object):

    def __init__(self, score_file) -> None:
        self.layer_configurations = []
        self.score_file = score_file
        self.samples = 0
        self.pertubations = 0
        self.classes = 0
        self.AOPC = 0
        self.title = ""

        self._infer_config_from_file_name(score_file)
        self._find_shapes(score_file)
        self._parse_file(score_file)

    def __str__(self) -> str:
        res = ""
        if self.layer_configurations:
            for config in self.layer_configurations:
                res += "{}\n".format(config)
        else:
            res += self.title + "\n"
        res += "{:12}: {}\n".format('File', re.sub(file_name_re, r'\1', self.score_file))
        res += "{:12}: {:10f}".format('AOPC', self.AOPC)
        return res

    def short_description(self) -> str:
        if self.title in ['Random', 'Sensitivity analysis']:
            return self.title
        else:
            return re.sub(file_name_re, r'\1', self.score_file)

    def get_AOPC(self) -> float:
        return self.AOPC

    def _infer_config_from_file_name(self, score_file) -> None:
        if 'random' in score_file:
            self.title = 'Random'
            return
        elif 'sensitivity' in score_file:
            self.title = 'Sensitivity analysis'
            return

        if "SM" in score_file:
            layer_dict = re.match(type_config_sm_re, score_file).groupdict()
        else:
            layer_dict = re.match(type_config_re, score_file[:-4]).groupdict()

        print(layer_dict)
        for (layer, conf_string) in layer_dict.items():
            if layer == 'sparse' or conf_string is None:
                continue
            self.layer_configurations.append(Config(layer, conf_string))
        self.layer_configurations.sort(key=lambda x: x.layer)

    def _find_shapes(self, score_file) -> None:
        with open(score_file, 'r') as sf:
            headings = sf.readline().split()
            column_count = len(headings)
            self.pertubations = column_count - 3

            sample_count = 0
            classes = 0
            state = True
            for line in sf:
                if len(line) < 10:
                    if state:
                        # Blank line
                        sample_count += 1
                        self.classes = classes
                        classes = 0
                        state = False
                else:
                    classes += 1
                    state = True
            self.samples = sample_count

    def _parse_file(self, score_file):
        labels = []
        predictions = []
        x0_predictions = []
        pertubation_scores = []

        with open(score_file, 'r') as sf:
            # Read headings
            sf.readline()
            for i in range(self.samples):
                predicted_class = 0
                line = sf.readline()
                split = line.split()
                labels.append(int(split[0]))
                predicted_class = int(split[1])
                predictions.append(predicted_class)
                
                x0_predictions.append(float(split[2]))
                pertubation_scores.append([float(k) for k in split[3:]])

                # Read blank line
                sf.readline()

        self.labels = np.array(labels)
        self.predictions = np.array(predictions)
        self.x0_predictions = np.array(x0_predictions)

        # Shape (samples, pertubations)
        self.pertubation_scores = np.array(pertubation_scores)

        if self.pertubation_scores.shape[1] == 99:
            self.pertubation_scores = np.pad(self.pertubation_scores, [[0, 0], [0, 1]], 'edge')
            self.pertubations += 1

        differences = self.x0_predictions.reshape((self.samples, 1)) - self.pertubation_scores

        self.AOPC = np.mean(np.sum(differences, 1)) / (1 + self.pertubations)

        self.pertubation_scores = np.mean(self.pertubation_scores, 0)
        self.x0_predictions = np.mean(self.x0_predictions, 0, keepdims=True)
        


