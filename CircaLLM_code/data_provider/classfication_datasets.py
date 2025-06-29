from pickle import NONE
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data import load_from_tsfile, MultiGroup_load_from_tsfile


class ClassificationDataset:

    def __init__(self, data_split="train",file_path="datasets/",seq_len=72):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """
        self.seq_len = seq_len
        self.train_file_path_and_name = file_path+"TRAIN.ts"
        self.val_file_path_and_name = file_path+"VALIDATION.ts"
        self.test_file_path_and_name = file_path+"TEST.ts"
        self.data_split = data_split  # 'train' or 'test'

        # Read data
        self._read_data()

    def _transform_labels(self, train_labels: np.ndarray,val_labels:np.ndarray, test_labels: np.ndarray):
        labels = np.unique(train_labels)  # Move the labels to {0, ..., L-1}
        
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        train_labels = np.vectorize(transform.get)(train_labels)
        val_labels = np.vectorize(transform.get)(val_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, val_labels, test_labels

    def __len__(self):
        return self.num_timeseries

    def _read_data(self):
        self.scaler = StandardScaler()

        self.train_data, self.train_labels = load_from_tsfile(self.train_file_path_and_name)
        self.val_data, self.val_labels = load_from_tsfile(self.val_file_path_and_name)
        self.test_data, self.test_labels = load_from_tsfile(self.test_file_path_and_name)

        self.train_labels, self.val_labels, self.test_labels = self._transform_labels(
            self.train_labels, self.val_labels, self.test_labels
        )

        if self.data_split == "train":
            self.data = self.train_data
            self.labels = self.train_labels
        elif self.data_split == "val":
            self.data = self.val_data
            self.labels = self.val_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

        self.num_timeseries = self.data.shape[0]
        self.len_timeseries = self.data.shape[2]

        self.data = self.data.reshape(-1, self.len_timeseries)
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.data = self.data.reshape(self.num_timeseries, self.len_timeseries)

        self.data = self.data.T

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[:, index]
        timeseries_len = len(timeseries)
        labels = self.labels[index,].astype(int)
        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        timeseries = np.pad(timeseries, (self.seq_len - timeseries_len, 0))

        return np.expand_dims(timeseries, axis=0), input_mask, labels

#Rhythm-Circadian
class DataSplit:
    def __init__(self,data,labels,time_stamp,seq_len=72):
        self.data=data
        self.labels=labels
        self.time_stamp=time_stamp
        self.seq_len=seq_len
        self._length=len(self.data)
        self.timesteps=self.data.shape[-1]


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        """
        Return one sample from the dataset.
        """
        assert index < self.__len__()
        # Access the data, labels, and mask for the chosen index
        labels = self.labels[index].astype(int)

        timeseries = self.data[index]
        timeseries_len = timeseries.shape[-1]
        print(f'timeseries:{timeseries}')
        print(f'timeseries.shape:{timeseries.shape}')
        # # Create input mask for padding
        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        # Pad the timeseries to the expected length
        # timeseries = np.pad(timeseries, ((0,0),(0,0),(self.seq_len - self.timesteps, 0)))
        timeseries = np.pad(timeseries, ((0,0),(self.seq_len - self.timesteps, 0)))

        x_mark = np.pad(self.time_stamp, ((0,0),(self.seq_len - self.timesteps, 0),(0,0)),constant_values=0)

        return timeseries, input_mask, x_mark[index],labels
    
class newClassificationDataset:
    def __init__(self,data_split=None,file_path="datasets/",seq_len=72,Realcase=False):
        """
        Initializes the dataset by loading train, validation, and test data.
        
        Parameters
        ----------
        file_path : str
            Path to the dataset.
        """
        self.Realcase=Realcase
        self.seq_len = seq_len
        self.train_file_path_and_name = file_path + "TRAIN.ts"
        self.val_file_path_and_name = file_path + "VALIDATION.ts"
        self.test_file_path_and_name = file_path + "TEST.ts"
        self.data_split = data_split
        self.singleDatapath=file_path+data_split.upper()+".ts" if data_split else None
        # Read all data
        self._read_data()

    def _transform_labels(self, train_labels, val_labels, test_labels):
        """
        Transform the labels to {0, ..., L-1} format.
        """
        labels = np.unique(train_labels)  # Move the labels to {0, ..., L-1}
        transform = {l: i for i, l in enumerate(labels)}

        train_labels = np.vectorize(transform.get)(train_labels)
        val_labels = np.vectorize(transform.get)(val_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, val_labels, test_labels
    @property
    def train_dat(self):
        """返回训练数据作为ClassificationDataset实例"""
        return self._get_data("train")

    @property
    def val_dat(self):
        """返回验证数据作为ClassificationDataset实例"""
        return self._get_data("val")

    @property
    def test_dat(self):
        """返回测试数据作为ClassificationDataset实例"""
        return self._get_data("test")
    @property
    def aper_dat(self):
        """返回测试数据作为ClassificationDataset实例"""
        return self._get_data("aper")
    
    def _read_data(self):
        """
        Read data from files and preprocess (scaling, reshaping, etc.).
        """
        if self.data_split:
            self.singleData, self.aperLabels, self.aper_time_stamp = load_from_tsfile(self.singleDatapath, return_meta_data=False, Realcase=self.Realcase)
            # self.singleData = self._process_data(self.singleData)
            # self._length=self.singleData.shape[0]
        else:
            # Load data from respective files
            self.train_data, self.train_labels, self.train_time_stamp = load_from_tsfile(self.train_file_path_and_name, return_meta_data=False, Realcase=self.Realcase)
            self.val_data, self.val_labels,  self.val_time_stamp = load_from_tsfile(self.val_file_path_and_name,return_meta_data=False,Realcase=self.Realcase)
            self.test_data, self.test_labels, self.test_time_stamp = load_from_tsfile(self.test_file_path_and_name,return_meta_data=False,Realcase=self.Realcase)
            # print(f'self.val_labels:{self.val_labels}')
            if self.Realcase==False:
                # Transform labels to a unified format
                self.train_labels, self.val_labels, self.test_labels = self._transform_labels(
                    self.train_labels, self.val_labels, self.test_labels
                )
            

                # self.train_labels, self.val_labels, self.test_labels = self._transform_labels(
                #     self.train_labels, self.val_labels, self.test_labels
                # )
            # Transpose data to shape [seq_len, num_timeseries] for easier indexing
            # self.train_data = self._process_data(self.train_data)
            # self.val_data = self._process_data(self.val_data)
            # self.test_data = self._process_data(self.test_data)
        
    def _process_data(self, data):
        # 数据预处理，标准化等
        n_samples, n_channels, timesteps = data.shape

        data = data.reshape(n_samples*n_channels,timesteps)
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        data = (data - mean) / (std+1e-6)
        
        # data = data.T
        # data = data.reshape(n_samples, timesteps) if n_channels==1 else data.reshape(n_samples, n_channels,timesteps)
        data=data.reshape(n_samples, n_channels,timesteps)
        return data
    
    def _get_data(self, split):
        """
        根据split返回对应的数据和标签
        """
        if split=="aper":
            data, labels, time_stamp = self.singleData, self.aperLabels, self.aper_time_stamp
        elif split == "train":
            data, labels, time_stamp = self.train_data, self.train_labels, self.train_time_stamp
        elif split == "val":
            data, labels, time_stamp = self.val_data, self.val_labels, self.val_time_stamp
        elif split == "test":
            data, labels, time_stamp = self.test_data, self.test_labels, self.test_time_stamp
        else:
            raise ValueError(f"Unknown split: {split}")

        # 创建并返回一个新的ClassificationDataset实例
        dataset_instance = DataSplit(data,labels,time_stamp,self.seq_len)
        return dataset_instance
    
class MultipleDataset(DataSplit):
    def __init__(self,data_split="train",file_paths=["datasets/"],seq_len=72,seed=123,logger=None,type="synth",Realcase=False): 
        self.data_split = data_split
        self.datasets = []
        self.seq_len = seq_len
        self.seed=seed
        self.type=type
        self.mask=np.ones(self.seq_len)
        self.logger=logger
        self.Realcase=Realcase
        if self.data_split=="aper":
            aperSplit="test"
        else :
            aperSplit=None
        i=0
        # 循环加载所有数据集
        self.logger.info(f"{data_split.upper()}") if self.logger else print(f"{data_split.upper()}")

        for file_path in file_paths:
            dataset = newClassificationDataset(file_path=file_path,seq_len=seq_len,data_split=aperSplit,Realcase=self.Realcase)
            i=i+1
            
            self.logger.info(f"    Loading {i}th dataset ...") if self.logger else print(f"    Loading {i}th dataset ...")
            self.datasets.append(dataset)

        self._merge_datasets()

    def _merge_datasets(self):
        """
        合并所有数据集的训练数据、标签和其他相关信息。
        """
        # 初始化一个空的容器用于存储所有的数据和标签
        all_data = []
        all_labels = []
        all_x_mark=[]
        # 循环每个数据集并合并数据
        for dataset in self.datasets:
            if self.data_split=="aper":
                split_data=dataset.aper_dat
            elif self.data_split=="train":
                split_data=dataset.train_dat
            elif self.data_split == "val":
                split_data=dataset.val_dat
            else:
                split_data=dataset.test_dat

            labels=split_data.labels
            data=split_data.data
            time_stamp=split_data.time_stamp

            x_mark = np.pad(time_stamp, ((0,0),(self.seq_len - split_data.timesteps, 0),(0,0)),constant_values=0)

            data = np.pad(data, ((0,0),(0,0),(self.seq_len - split_data.timesteps, 0)))

            all_data.append(data)
            all_x_mark.append(x_mark)
            all_labels.append(labels)
            

        self.timesteps=split_data.timesteps
        # 将所有数据拼接成一个大的数组
        all_data = np.concatenate(all_data, axis=0)  # 按行拼接数据
        all_labels = np.concatenate(all_labels, axis=0)  # 按行拼接标签
        all_x_mark = np.concatenate(all_x_mark, axis=0)  # 按行拼接标签
        # 打乱数据和标签的顺序
        indices = np.arange(all_data.shape[0])  # 获取样本索引
        np.random.seed(self.seed)
        np.random.shuffle(indices)  # 打乱索引顺序

        # 使用打乱后的索引来重新排序数据和标签
        self.data = all_data[indices]  # 根据打乱的索引重新排序数据
        self.labels = all_labels[indices]  # 根据打乱的索引重新排序标签
        self.x_mark = all_x_mark[indices]  # 根据打乱的索引重新排序标签 
        self.num_timeseries = self.data.shape[0]  # 合并后的时间序列个数
        # self.timesteps = self.data.shape[-1]  # 每个时间序列的长度

    def __len__(self):
        return self.num_timeseries
        
    def __getitem__(self, index):
        """
        Return one sample from the dataset.
        """
        
        assert index < self.__len__()
        # Access the data, labels, and mask for the chosen index
        labels = self.labels[index].astype(int)

        timeseries = self.data[index]
        x_mark = self.x_mark[index]
        # Create input mask for padding
        input_mask = np.ones(self.seq_len)

        input_mask[: self.seq_len - self.timesteps] = 0
        # Pad the timeseries to the expected length
        # timeseries = np.pad(timeseries, ((0,0),(self.seq_len - self.timesteps, 0)))
        return timeseries, input_mask, x_mark, labels



#diff-Circadian
class MulGroup_DataSplit:
    def __init__(self,data_1, time_stamp_1, data_2, time_stamp_2, labels, seq_len=72):
        self.data_1, self.data_2=data_1, data_2
        self.time_stamp_1, self.time_stamp_2=time_stamp_1, time_stamp_2
    
        self.labels=labels
        self.seq_len=seq_len

        self._length_1, self._length_2=len(self.data_1), len(self.data_2)

        self.timesteps_1, self.timesteps_2=self.data_1.shape[-1], self.data_2.shape[-1]
    def __len__(self):
        if self._length_1==self._length_2:
            return self._length_1
        else:
            return -1

    def __getitem__(self, index):
        """
        Return one sample from the dataset.
        """

        assert index < self.__len__()
        # Access the data, labels, and mask for the chosen index
        labels = self.labels[index].astype(int)

        timeseries_1, timeseries_2 = self.data_1[index], self.data_2[index]
        timeseries_len_1, timeseries_len_2 = timeseries_1.shape[-1], timeseries_2.shape[-1]

        # # Create input mask for padding
        input_mask_1,input_mask_2 = np.ones(self.seq_len),np.ones(self.seq_len)
        input_mask_1[: self.seq_len - timeseries_len_1],input_mask_2[: self.seq_len - timeseries_len_2] = 0,0
        
        # Pad the timeseries to the expected length
        timeseries_1 = np.pad(timeseries_1, ((0,0),(self.seq_len - self.timesteps_1, 0)))
        timeseries_2 = np.pad(timeseries_2, ((0,0),(self.seq_len - self.timesteps_2, 0)))

        x_mark_1 = np.pad(self.time_stamp_1, ((0,0),(self.seq_len - self.timesteps_1, 0),(0,0)),constant_values=0)
        x_mark_2 = np.pad(self.time_stamp_2, ((0,0),(self.seq_len - self.timesteps_2, 0),(0,0)),constant_values=0)


        return timeseries_1, input_mask_1, x_mark_1[index], timeseries_2, input_mask_2, x_mark_2[index], labels
    
class MulGroup_ClassificationDataset:
    def __init__(self,data_split=None,file_path="datasets/",seq_len=72,Realcase=False):
        """
        Initializes the dataset by loading train, validation, and test data.
        
        Parameters
        ----------
        file_path : str
            Path to the dataset.
        """
        self.seq_len = seq_len
        self.data_split = data_split
        self.Realcase=Realcase
        self.train_file_path_and_name = file_path + "TRAIN.ts"
        self.val_file_path_and_name = file_path + "VALIDATION.ts"
        self.test_file_path_and_name = file_path + "TEST.ts"
        
        # Read all data
        self._read_data()
    
    def _read_data(self):
        """
        Read data from files and preprocess (scaling, reshaping, etc.).
        """
        # Load data from respective files
        if self.data_split == 'test':
            self.test_data_1, self.test_time_stamp_1, self.test_data_2, self.test_time_stamp_2, self.test_labels = MultiGroup_load_from_tsfile(self.test_file_path_and_name,return_meta_data=False,Realcase=self.Realcase)
        else:
            self.train_data_1, self.train_time_stamp_1, self.train_data_2, self.train_time_stamp_2, self.train_labels = MultiGroup_load_from_tsfile(self.train_file_path_and_name,return_meta_data=False,Realcase=self.Realcase)
            self.val_data_1, self.val_time_stamp_1, self.val_data_2, self.val_time_stamp_2, self.val_labels = MultiGroup_load_from_tsfile(self.val_file_path_and_name,return_meta_data=False,Realcase=self.Realcase)
            self.test_data_1, self.test_time_stamp_1, self.test_data_2, self.test_time_stamp_2, self.test_labels = MultiGroup_load_from_tsfile(self.test_file_path_and_name,return_meta_data=False,Realcase=self.Realcase)

    def _get_data(self, split):
        """
        根据split返回对应的数据和标签
        """
        if split == "train":
            data_1, time_stamp_1, data_2, time_stamp_2, labels = self.train_data_1, self.train_time_stamp_1, self.train_data_2, self.train_time_stamp_2, self.train_labels
        elif split == "val":
            data_1, time_stamp_1, data_2, time_stamp_2, labels = self.val_data_1, self.val_time_stamp_1, self.val_data_2, self.val_time_stamp_2, self.val_labels
        elif split == "test":
            data_1, time_stamp_1, data_2, time_stamp_2, labels = self.test_data_1, self.test_time_stamp_1, self.test_data_2, self.test_time_stamp_2, self.test_labels
        else:
            raise ValueError(f"Unknown split: {split}")

        # 创建并返回一个新的ClassificationDataset实例
        dataset_instance = MulGroup_DataSplit(data_1, time_stamp_1, data_2, time_stamp_2, labels, self.seq_len)
        return dataset_instance
    
    @property
    def train_dat(self):
        """返回训练数据作为ClassificationDataset实例"""
        return self._get_data("train")

    @property
    def val_dat(self):
        """返回验证数据作为ClassificationDataset实例"""
        return self._get_data("val")

    @property
    def test_dat(self):
        """返回测试数据作为ClassificationDataset实例"""
        return self._get_data("test")

class MulGroup_MultipleDataset(MulGroup_DataSplit):
    def __init__(self,data_split="train",file_paths=["datasets/"],seq_len=72,seed=123,logger=None,Realcase=False): 
        self.data_split = data_split
        self.datasets = []
        self.seq_len = seq_len
        self.seed=seed
        self.logger=logger
        self.Realcase=Realcase
        i=0
        # 循环加载所有数据集
        self.logger.info(f"{data_split.upper()}") if self.logger else print(f"{data_split.upper()}")

        for file_path in file_paths:
            dataset = MulGroup_ClassificationDataset(file_path=file_path,seq_len=seq_len,data_split=self.data_split,Realcase=self.Realcase)
            i=i+1
            
            self.logger.info(f"    Loading {i}th dataset ...") if self.logger else print(f"    Loading {i}th dataset ...")
            self.datasets.append(dataset)

        self._merge_datasets()
    def _merge_datasets(self):
        """
        合并所有数据集的训练数据、标签和其他相关信息。
        """
        # 初始化一个空的容器用于存储所有的数据和标签
        all_data_1, all_data_2 = [],[]
        all_labels = []
        all_x_mark_1, all_x_mark_2=[],[]
        # 循环每个数据集并合并数据
        for dataset in self.datasets:
            if self.data_split=="train":
                split_data=dataset.train_dat
            elif self.data_split == "val":
                split_data=dataset.val_dat
            else:
                split_data=dataset.test_dat
        
            labels=split_data.labels
            data_1, data_2=split_data.data_1, split_data.data_2
            time_stamp_1, time_stamp_2=split_data.time_stamp_1, split_data.time_stamp_2

            x_mark_1 = np.pad(time_stamp_1, ((0,0),(self.seq_len - split_data.timesteps_1, 0),(0,0)),constant_values=0)
            x_mark_2 = np.pad(time_stamp_2, ((0,0),(self.seq_len - split_data.timesteps_2, 0),(0,0)),constant_values=0)

            data_1 = np.pad(data_1, ((0,0),(0,0),(self.seq_len - split_data.timesteps_1, 0)))
            data_2 = np.pad(data_2, ((0,0),(0,0),(self.seq_len - split_data.timesteps_2, 0)))

            all_data_1.append(data_1)
            all_data_2.append(data_2)
            all_x_mark_1.append(x_mark_1)
            all_x_mark_2.append(x_mark_2)
            all_labels.append(labels)
            
        self.timesteps_1, self.timesteps_2=split_data.timesteps_1, split_data.timesteps_2
        # 将所有数据拼接成一个大的数组
        all_data_1, all_data_2 = np.concatenate(all_data_1, axis=0), np.concatenate(all_data_2, axis=0) # 按行拼接数据
        all_x_mark_1, all_x_mark_2 = np.concatenate(all_x_mark_1, axis=0), np.concatenate(all_x_mark_2, axis=0)  # 按行拼接标签
        all_labels = np.concatenate(all_labels, axis=0)  # 按行拼接标签

        # 打乱数据和标签的顺序
        indices = np.arange(all_data_1.shape[0])  # 获取样本索引
        np.random.seed(self.seed)
        np.random.shuffle(indices)  # 打乱索引顺序

        # 使用打乱后的索引来重新排序数据和标签
        self.data_1,self.data_2 = all_data_1[indices], all_data_2[indices]  # 根据打乱的索引重新排序数据
        self.x_mark_1,self.x_mark_2 = all_x_mark_1[indices], all_x_mark_2[indices]  # 根据打乱的索引重新排序标签 
        self.labels = all_labels[indices]  # 根据打乱的索引重新排序标签

        self.num_timeseries_1, self.num_timeseries_2 = self.data_1.shape[0], self.data_2.shape[0]  # 合并后的时间序列个数
        # self.timesteps = self.data.shape[-1]  # 每个时间序列的长度
    def __len__(self):
        if self.num_timeseries_1==self.num_timeseries_2:
            return self.num_timeseries_1
        else:
            return -1
    def __getitem__(self, index):
        """
        Return one sample from the dataset.
        """
        assert index < self.__len__()
        
        # Access the data, labels, and mask for the chosen index
        labels = self.labels[index].astype(int)

        timeseries_1, timeseries_2 = self.data_1[index], self.data_2[index]
        x_mark_1, x_mark_2 = self.x_mark_1[index], self.x_mark_2[index]
        
        # Create input mask for padding
        input_mask_1, input_mask_2 = np.ones(self.seq_len), np.ones(self.seq_len)

        input_mask_1[: self.seq_len - self.timesteps_1] = 0
        input_mask_2[: self.seq_len - self.timesteps_2] = 0
        # Pad the timeseries to the expected length
        # timeseries = np.pad(timeseries, ((0,0),(self.seq_len - self.timesteps, 0)))
        return timeseries_1, input_mask_1, x_mark_1, timeseries_2, input_mask_2, x_mark_2, labels