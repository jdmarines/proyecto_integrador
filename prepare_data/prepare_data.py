import tensorflow as tf


class LoadData:
    def __init__(self, path, sep = '|', *args, **kwargs):
        self.path = path
        self.args = args
        self.sep = sep
        self.kwargs = kwargs
        self.dataset = self.load_and_parse()

    def load_and_parse(self):
        data = tf.data.TextLineDataset(self.path)
        data = data.map(self.parse_data)
        return data

    def parse_data(self, line):
        parsed_data = {}
        get_lines = tf.strings.strip(line)
        get_fields = tf.strings.split(get_lines, sep = self.sep)
        for column_name, indexes in self.kwargs.items():
            parsed_data[column_name] = get_fields[indexes]
        return parsed_data
        



        



