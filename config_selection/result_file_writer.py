import os
from config_selection import logger

class ResultWriter(object):
    def __init__(self, dest) -> None:
        self._destination_folder = dest[:-1] if dest[-1] in ['/', '\\'] else dest

        i = 1
        destination = self._destination_folder
        while os.path.exists(destination):
            destination = "{}_{:02}".format(self._destination_folder, i)
            i += 1

        self._destination_folder = destination
        logger.info( "Writing results to {}".format(self._destination_folder))
        os.makedirs(self._destination_folder)

    def write_result(self, config, label, prediction, results):
        file_name = "{}/{}.res".format(self._destination_folder, config)
        exists = os.path.isfile(file_name)

        with open(file_name, 'a') as f:
            if not exists:
                num_iterations = results.shape[2]
                str_format = "{:10} {:10} " + "{:>10} " * num_iterations
                headings = str_format.format('label', 'pred', *[i for i in range(num_iterations)]) + '\n'
                f.write(headings)

            for i, sample in enumerate(results):
                for row in sample:
                    fmt = "{:>10d} " * 2
                    fmt += "{:>10.6f} " * len(row)
                    res = fmt.format(label[i], prediction[i], *row) + "\n"
                    f.write(res)
                f.write("\n")

        logger.info("Saved result for config {}".format(config))

