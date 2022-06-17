import os

from torch.utils.tensorboard import SummaryWriter


class CSVWriter(SummaryWriter):
    """
    Wrap Tensorboard.SummaryWriter with csv logger.
    """

    def __init__(self, log_path):

        super().__init__(log_path)

        self._log_path = log_path
        filename = os.path.join(log_path, 'progress.csv')
        self._csv_file = open(filename, 'w+t')
        self._keys = []
        self._sep = ','
        self._kvs = {}

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super().add_scalar(tag, scalar_value, global_step, walltime)
        if global_step is None:
            return
        if 'global_step' not in self._kvs:
            self._kvs['global_step'] = global_step
        if self._kvs['global_step'] != global_step:
            self._write_kvs(self._kvs)
        self._kvs['global_step'] = global_step
        self._kvs[tag] = scalar_value

    def _write_kvs(self, kvs):
        extra_keys = list(kvs.keys() - self._keys)
        extra_keys.sort()
        if extra_keys:
            self._keys.extend(extra_keys)
            self._csv_file.seek(0)
            lines = self._csv_file.readlines()
            self._csv_file.seek(0)
            for (i, k) in enumerate(self._keys):
                if i > 0:
                    self._csv_file.write(',')
                self._csv_file.write(k)
            self._csv_file.write('\n')
            for line in lines[1:]:
                self._csv_file.write(line[:-1])
                self._csv_file.write(self._sep * len(extra_keys))
                self._csv_file.write('\n')
        for (i, k) in enumerate(self._keys):
            if i > 0:
                self._csv_file.write(',')
            v = kvs.get(k)
            if v is not None:
                self._csv_file.write(str(v))
        self._csv_file.write('\n')
        self._csv_file.flush()

