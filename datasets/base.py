import abc


class BaseDataset(abc.ABC):
    @abc.abstractmethod
    def get_train_loader(self, batch_size: int = 8,num_workers:int=0,shuffle:bool=True, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_val_loader(self, batch_size: int = 8, num_workers:int=0,shuffle:bool=True, **kwargs):
        raise NotImplementedError
