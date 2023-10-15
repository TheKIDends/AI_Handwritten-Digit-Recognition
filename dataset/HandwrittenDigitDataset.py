from torch.utils.data import Dataset


class HandwrittenDigitDataset(Dataset):
    def __init__(self, input_arr, label_arr=None):
        self.input_arr = input_arr
        self.label_arr = label_arr

    def __len__(self):
        return len(self.input_arr)

    def __getitem__(self, id):
        if self.label_arr is None:
            return self.input_arr[id] / 255.
        return self.input_arr[id] / 255., self.label_arr[id]
