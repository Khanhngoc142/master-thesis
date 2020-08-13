import os
import random
import warnings

from extractor.crohme_parser.inkml import Ink
from extractor.crohme_parser.extract import Extractor
from utils.fs import get_source_root


class Symbol(object):
    def __init__(self, label):
        """
        Symbol represent a symbol with:
        - _label: Label of the symbol
        - _traces_archive: store traces (list of coords) .Represent handwritten styles/ instances
        :param label: label of symbol
        """
        self._label = label
        self._traces_archive = []
        self._archive_size = 0

    @property
    def label(self):
        return self._label

    @property
    def traces_archive(self):
        return self._traces_archive

    @property
    def archive_size(self):
        return self._archive_size

    @property
    def _random_traces(self):
        random_idx = random.randint(0, self._archive_size - 1)
        return random_idx, self._traces_archive[random_idx]

    def get_traces(self, i=None, fault_tolerance=True):
        """
        get a set of traces
        :param i: index to give. Optional. Default to be random.
        :param fault_tolerance: whether to raise IndexError. When False, get a random set of traces instead. Default: False.
        :return: (i: int, coords: list<list<[int, int]>>)
        """
        if i is None:
            return self._random_traces
        if i < self._archive_size:
            return i, self._traces_archive[i]
        else:
            warnings.warn("Request traces at index {} failed. "
                          "Archive for symbol {} "
                          "only has {} sample(s).".format(i, self._label, self._archive_size) +
                          (" Sample a random." if fault_tolerance else ""))
            if fault_tolerance:
                return self._random_traces
            else:
                raise IndexError("Invalid index: {}".format(i))

    def add_traces(self, traces):
        """
        add traces coords
        :param traces:
        :return:
        """
        self._traces_archive.append(traces)
        self._archive_size += 1
        return self._archive_size - 1


class Library(object):
    def __init__(self):
        self._lib = {}

    @property
    def symbols(self):
        return list(self._lib.keys())

    def get_symbol(self, sym):
        """
        get symbol sym
        :param sym: symbol label to get
        :return: Symbol
        """
        if sym in self._lib.keys():
            return self._lib[sym]
        else:
            raise ValueError("'{}' symbol not found in Library.".format(sym))

    def add_traces_to_symbol(self, sym, traces):
        """
        add a set of traces to symbol sym
        :param sym: symbol label to add
        :param traces: set of traces to add to symbol
        :return: Symbol
        """
        if sym in self._lib.keys():
            symbol = self._lib[sym]
        else:
            symbol = Symbol(sym)
            self._lib[sym] = symbol
        symbol.add_traces(traces)
        return symbol

    def get_traces_of_symbol(self, sym, i=None, fault_tolerance=True):
        """
        get a set of traces of specified symbol
        :param sym:
        :param i:
        :param fault_tolerance:
        :return: (i: int, traces: list<inkml.Trace>)
        """
        if sym in self._lib.keys():
            return self._lib[sym].get_traces(i, fault_tolerance)
        else:
            raise ValueError("'{}' symbol not found in Library.".format(sym))

    def extract_from_ink(self, ink: Ink, target_range=(0, 1)):
        for group in ink.trace_groups:
            lbl = group.label
            # compute scale
            (xmin, ymin), (xmax, ymax) = group.bbox
            xdiff = xmax - xmin
            ydiff = ymax - ymin
            diff = max(xdiff, ydiff)
            target_diff = target_range[1] - target_range[0]
            scale = target_diff / diff

            # compute new x, y origin
            xorigin = (target_range[1] - xdiff * scale) / 2
            yorigin = (target_range[1] - ydiff * scale) / 2
            group_traces = []
            for trace in group.traces:
                coords = [[xorigin + (coord[0] - xmin) * scale, yorigin + (coord[1] - ymin) * scale]
                          for coord in trace.coords]
                group_traces.append(coords)
            self.add_traces_to_symbol(lbl, group_traces)


def build_library(data_versions="2013", crohme_package=os.path.join("data", "CROHME_full_v2"), datasets="train",
                  target_coord_range=(0, 1)):
    crohme_package = os.path.join(get_source_root(), crohme_package)
    lib = Library()
    extractor = Extractor(data_versions, crohme_package)
    for ink in extractor.parse_inkmls_iterator(datasets=datasets):
        lib.extract_from_ink(ink, target_coord_range)
    return lib


if __name__ == "__main__":
    build_library()
