import logging
from time import localtime, strftime
import os.path

class TimeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        filename = filename.format(strftime("%Y-%m-%d_%H-%M", localtime()))

        if os.path.isfile(filename):
            with open(filename, 'a') as f:
                f.writelines(['\n', '####### starting new recording ######', '\n'])

        print('Logging to file: {}'.format(filename))
        super().__init__(filename, mode, encoding, delay)
