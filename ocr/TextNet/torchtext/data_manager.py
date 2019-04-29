from torch.utils.data import DataLoader

from .dataset_loader import DetectDataset
from .datasets import init_detect_dataset
from .transforms import build_transforms
from .samplers import build_train_sampler


class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root='./data',
                 height=672,
                 width=512,
                 train_batch_size=4,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        # self.random_erase = random_erase
        # self.color_jitter = color_jitter
        # self.color_aug = color_aug
        # self.num_instances = num_instances

        transform_train = build_transforms(
            self.height, self.width, batch_size=train_batch_size, is_train=True
        )
        transform_test = build_transforms(
            self.height, self.width, batch_size=train_batch_size, is_train=False
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader


class DetectImageManager(BaseDataManager):
    def __init__(self,
                 model,
                 use_gpu,
                 source_names,
                 target_names,
                 **kwargs
                 ):
        super(DetectImageManager, self).__init__(
            use_gpu, source_names, target_names, **kwargs)

        train = []
        for name in self.source_names:
            dataset = init_detect_dataset(root=self.root, name=name)

            for img_path, annotation_path, parse_txt in dataset.train:
                train.append([img_path, annotation_path, parse_txt])

        self.train_sampler = build_train_sampler(
            train, self.train_sampler
        )
        self.trainloader = DataLoader(
            DetectDataset(model, train, transform_image=self.transform_train), sampler=self.train_sampler,
            batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

        print('\n')
        print('  **************** Summary ****************')
        print('  train names      : {}'.format(self.source_names))
        print('  # train datasets : {}'.format(len(self.source_names)))
        print('  # train images   : {}'.format(len(train)))
        # print('  test names       : {}'.format(self.target_names))
        print('  *****************************************')
        print('\n')
