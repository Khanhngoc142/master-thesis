import os
from .inkml import parse_inkml_dir


class Extractor(object):
    _versions_available = ['2013', '2012', '2011']

    @property
    def versions_available(self):
        return self._versions_available

    def __init__(self, data_versions="2013", crohme_package=os.path.join("data", "CROHME_full_v2")):
        versions = data_versions.split('+')
        for ver in versions:
            if ver not in self._versions_available:
                raise ValueError("version {} not available".format(ver))
        self._versions = versions
        self._crohme_package = crohme_package
        self._setup_data_paths()

    def _setup_data_paths(self):
        self._train_dirs = []
        self._validation_dirs = []
        self._test_dirs = []

        for version in self._versions:
            train_dir = []
            test_dir = []
            validation_dir = []

            if version == "2011":
                data_dir = os.path.join(self._crohme_package, "CROHME2011_data")
                train_dir = [os.path.join(data_dir, "CROHME_training")]
                test_dir = [os.path.join(data_dir, "CROHME_testGT")]
                validation_dir = [os.path.join(data_dir, "CROHME_test")]

            if version == "2012":
                data_dir = os.path.join(self._crohme_package, "CROHME2012_data")
                train_dir = [os.path.join(data_dir, "trainData")]
                test_dir = [os.path.join(data_dir, "testDataGT")]
                validation_dir = [os.path.join(data_dir, "testData")]

            if version == "2013":
                data_dir = os.path.join(self._crohme_package, "CROHME2013_data")
                train_root_dir = os.path.join(data_dir, "TrainINKML")
                train_dir = []
                for subdir in ["expressmatch", "extension", "HAMEX", "KAIST", "MathBrush", "MfrDB"]:
                    train_dir.append(os.path.join(train_root_dir, subdir))
                test_dir = [os.path.join(data_dir, "TestINKML")]
                validation_dir = [os.path.join(data_dir, "TestINKMLGT")]

            self._train_dirs += train_dir
            self._test_dirs += test_dir
            self._validation_dirs += validation_dir

    def parse_inkmls_iterator(self, datasets="train"):
        for dataset in datasets.split('+'):
            if dataset not in ["train", "valid", "test"]:
                raise ValueError("dataset {} not found.".format(dataset))
        for dataset in datasets.split('+'):
            if dataset == "train":
                dirs = self._train_dirs
            elif dataset == "valid":
                dirs = self._validation_dirs
            elif dataset == "test":
                dirs = self._test_dirs
            else:
                dirs = None

            for directory in dirs:
                for ink in parse_inkml_dir(directory):
                    yield ink
