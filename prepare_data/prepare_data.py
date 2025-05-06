from prepare_data.methods.base import LoadDataset


class LoadData:
    def __init__(self,
                 load_strategy: LoadDataset):
        self.load_strategy = load_strategy

    def load_data(self):
        return self.load_strategy.load_data()

        



