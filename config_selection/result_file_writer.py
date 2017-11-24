import os
from config_selection import logger
import numpy as np

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

        label = label.reshape((label.shape[0], 1))
        prediction = prediction.reshape((prediction.shape[0], 1))
        ls = np.concatenate([label, prediction], axis=1)
        fmt = "{:>10d} " * 2 + "{:>10.6f} " * results.shape[2]

        with open(file_name, 'a') as f:
            if not exists:
                num_iterations = results.shape[2]
                str_format = "{:10} {:10} " + "{:>10} " * num_iterations
                headings = str_format.format('label', 'pred', *[i for i in range(num_iterations)]) + '\n'
                f.write(headings)

            for i, sample in enumerate(results):
                for row in sample:
                    f.write(fmt.format(*ls[i], *row) + "\n")
                f.write("\n")

        logger.info("Saved result for config {}".format(config))

    def write_input(self, X):
        self._write_sparse_tensor('inputs', X)
        logger.info("Saved input batch")

    def write_explanation(self, config, R):
        self._write_sparse_tensor('rel_{}'.format(config), R)
        logger.info("Saved relevances for batch")

    def _clean_sparse_values(self, sparse_tensor):
        sel = np.where(sparse_tensor.values > 1e-8)
        return sparse_tensor.indices[sel], sparse_tensor.values[sel]

    def _write_sparse_tensor(self, file_name, sparse_tensor):
        file_name = "{}/{}.spt".format(self._destination_folder, file_name)
        indices, values = self._clean_sparse_values(sparse_tensor)

        with open(file_name, 'a') as f:
            # Write shape
            f.write("{} {} {}\n\n".format(*sparse_tensor.dense_shape))

            # Write indices
            indices = indices.transpose()
            num_values = indices.shape[1]
            fmt = "{:>10} " * num_values + "\n"
            for dimension in indices:
                f.write(fmt.format(*dimension))

            # Write values
            fmt = "\n" + "{:>10.8f} " * num_values + "\n\n\n"
            f.write(fmt.format(*values))

