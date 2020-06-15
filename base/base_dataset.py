import torch
import logging
import gc
from abc import ABCMeta, abstractmethod
from typing import Callable, List
from torch.utils.data import Dataset
from torch_geometric.data import Data
import random

class BaseDataSet(Dataset):

    def __init__(self,  root_dir: str,
                 start_level: int,
                 end_level: int,
                 get_coords: bool = False,
                 benchmark: bool = False,
                 is_train: bool = True,
                 debug_mode: bool = False,
                 transform: List[Callable] = None,
                 original_meshes_dir: str = None,
                 sample_checker: List[Callable] = None,
                 include_edges: bool = True):
        """Base class for data set objects.

        Arguments:
            root_dir {str} -- Root path to folder containing train/test samples
            start_level {int} -- First hierarchy level of mesh
            end_level {int} -- Final level of mesh

        Keyword Arguments:
            get_coords {bool} -- load coordinates of vertices as well (default: {False})
            benchmark {bool} -- benchmark mode (e.g. ScanNet test set) (default: {False})
            is_train {bool} -- train or validation mode? (default: {True})
            debug_mode {bool} -- also load vertex positions and colors of later levels (default: {False})
            transform {List[Callable]} -- list of callable objects to transform data samples (data augmentation) (default: {None})
            original_meshes_dir {str} -- root path of folder containing the original data set meshes (visualization and evaluation) (default: {None})
            sample_checker {List[Callable]} -- list of callable objects to reject samples of low quality (e.g. number of unlabeled vertices) (default: {None})
            include_edges {bool} -- only use point cloud information (default: {True})
        """

        self._root_dir = root_dir
        self._transform = transform
        self._start_level = start_level
        self._end_level = end_level
        self._debug_mode = debug_mode
        self._is_train = is_train
        self._original_meshes_dir = original_meshes_dir
        self._sample_checker = sample_checker
        self._benchmark = benchmark
        self._include_edges = include_edges

        self._get_coords = get_coords

        if self._sample_checker is None:
            self._sample_checker = [lambda x: True]
            
        self.index2filenames = self._load(self._is_train, self._benchmark)

        self.logger = logging.getLogger(self.__class__.__name__)

    def _load(self, is_train: bool, benchmark: bool) -> List[str]:
        """Subclasses implement function which returns file names of train/val samples
        """
        raise NotImplementedError("")

    def __getitem__(self, index: int) -> List[Data]:
        """[summary]

        Arguments:
            index {int} -- [description]

        Raises:
            RuntimeError: [description]

        Returns:
            List[Data] -- [description]
        """
        sample = None

        try:
            name = self.index2filenames[index]

            file_path = f"{self._root_dir}/{name}"

            saved_tensors = torch.load(file_path)

            coords = saved_tensors['vertices'][:self._end_level]

            if not self._benchmark:
                labels = saved_tensors['labels']
            else:
                labels = None

            edges = saved_tensors['edges'][:self._end_level]

            if self._is_train:
                traces = saved_tensors['traces'][:self._end_level-1]
            else:
                trace_0 = saved_tensors['traces'][0]
                traces = saved_tensors['traces'][1:self._end_level]

            sample = Data(x=coords[0][:, 3:],
                          pos=coords[0][:, :3],
                          edge_index=edges[0].t().contiguous(
            ) if self._include_edges else None,
                y=labels)
            sample.name = name

            nested_meshes = []

            for level in range(1, len(edges)):
                data = Data(edge_index=edges[level].t(
                ).contiguous() if self._include_edges else None)

                data.trace_index = traces[level-1]

                if self._debug_mode:
                    data.x = coords[level][:, 3:]
                    data.pos = coords[level][:, :3]

                if self._get_coords:
                    data.pos = coords[level][:, :3]

                nested_meshes.append(data)

            sample.num_vertices = []
            for level, nested_mesh in enumerate(nested_meshes):
                setattr(
                    sample, f"hierarchy_edge_index_{level+1}", nested_mesh.edge_index)
                setattr(
                    sample, f"hierarchy_trace_index_{level+1}", nested_mesh.trace_index)

                sample.num_vertices.append(
                    int(sample[f"hierarchy_trace_index_{level+1}"].max() + 1))

                if self._get_coords:
                    setattr(sample, f"pos_{level + 1}", nested_mesh.pos)

            if not self._is_train:
                sample.original_index_traces = trace_0

            if self._debug_mode:
                sample.all_levels = [sample]
                sample.all_levels.extend(nested_meshes)

            if self._transform:
                sample = self._transform(sample)

            for checker in self._sample_checker:
                if not checker(sample):
                    raise RuntimeError(
                        f"{checker.__class__.__name__} rejected the sample")

            return sample
        except Exception as e:
            if sample is not None:
                del sample
                gc.collect()
            self.logger.warning(
                f"Warning: Training example {index} could not be processed")
            self.logger.warning(f"{str(e)}")
            index = random.randrange(len(self))
            return self.__getitem__(index)

    def __len__(self):
        return len(self.index2filenames)

    @property
    @abstractmethod
    def color_map(self):
        """Dataset has to declare to which color each class is mapped for visualization purposes
        """
        raise NotImplementedError("")

    @property
    def num_classes(self):
        return len(self.color_map)

    @property
    @abstractmethod
    def ignore_classes(self) -> int:
        """There exist unlimited vertices. Specify ID to ignore them in calculation of loss and metrics."""
        raise NotImplementedError("")

    pos_neg_map = torch.FloatTensor(
        [
            [200, 200, 200],
            [0, 255, 0],
            [255, 0, 0]])
