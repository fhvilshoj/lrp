import os
from config_selection import logger

class ResultWriter(object):
    def __init__(self, dest) -> None:
        self._destination_folder = dest[:-1] if dest[-1] in ['/', '\\'] else dest
        if not os.path.exists(self._destination_folder):
            logger.info("Result path didn't exist. Creating path: {}".format(self._destination_folder))
            os.makedirs(self._destination_folder)

    def write_result(self, config, label, prediction, results):
        file_name = "{}/{}.res".format(self._destination_folder, config)
        exists = os.path.isfile(file_name)

        with open(file_name, 'a') as f:
            if not exists:
                num_iterations = results.shape[1]
                str_format = "{:10} {:10} " + "{:>10} " * num_iterations
                headings = str_format.format('label', 'pred', *[i for i in range(num_iterations)]) + '\n'
                f.write(headings)
            else:
                f.write("\n")

            for row in results:
                fmt = "{:>10d} " * 2
                fmt += "{:>10.6f} " * len(row)
                res = fmt.format(label.item(0), prediction.item(0), *row) + "\n"
                f.write(res)

        logger.info("Saved result for config {}".format(config))

