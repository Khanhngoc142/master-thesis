import os
import random
import warnings

from extractor.crohme_parser.inkml import Ink
from extractor.crohme_parser.extract import Extractor
from utils.fs import get_source_root
from utils.image_processing import get_trace_group_bbox, shift_trace_group_coords, centerize_trace_group_coords, \
    scale_trace_group, scale2height_trace_group, scale2width_trace_group, scale_trace_group_v2, scale_equation, get_equation_bbox_size, shift_equation, get_equation_bbox
from extractor.crohme_parser.fonts import StandardFont


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

    def get_sample(self, i=None, fault_tolerance=True):
        """
        get a sample of traces
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

    def add_sample(self, traces):
        """
        add a sample for symbol
        :param traces: a list of traces coordinates, 3 layer list
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

    def add_sample_to_symbol(self, sym, traces):
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
        symbol.add_sample(traces)
        return symbol

    def get_sample_of_symbol(self, sym, i=None, fault_tolerance=True):
        """
        get a sample of traces of specified symbol
        :param sym: symbol to get
        :param i: index of sample to get
        :param fault_tolerance:
        :return: (i: int, traces: 3-layer list)
        """
        if sym in self._lib.keys():
            return self._lib[sym].get_sample(i, fault_tolerance)
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
            self.add_sample_to_symbol(lbl, group_traces)

    def generate_equation_traces(self, equation):
        """
        Generate euation traces from random sample
        :param equation:
        :return: list of list of traces
        """
        gen_equation = []
        last_symbol_xmax = None
        subscript_flag = superscript_flag = False
        for e in equation:
            if isinstance(e, str):
                if e == '^':
                    superscript_flag = True
                    continue
                elif e == '_':
                    subscript_flag = True
                    continue
                _, cur_symbol = self.get_sample_of_symbol(e)
                cur_scale, cur_yshift = StandardFont.font_metric[e]
                if cur_scale != 1:
                    if e in '-+=':  # special characters need to be handle differently
                        _, cur_symbol = scale2width_trace_group(cur_symbol, cur_scale)
                    elif e in ',.':
                        _, cur_symbol = scale_trace_group_v2(cur_symbol, cur_scale)
                    else:
                        _, cur_symbol = scale2height_trace_group(cur_symbol, cur_scale)
                xrandom, yrandom = 0.2, 0  # TODO: implement some random shift
                if last_symbol_xmax is not None:
                    xshift = last_symbol_xmax
                else:
                    xshift = 0
                cur_symbol = shift_trace_group_coords(cur_symbol, xshift=xshift + xrandom, yshift=cur_yshift + yrandom)
                last_symbol_xmax = get_trace_group_bbox(cur_symbol)[2]
                gen_equation.append(cur_symbol)
            elif isinstance(e, list):
                child_eq = self.generate_equation_traces(e)
                if subscript_flag or superscript_flag:
                    child_eq = scale_equation(child_eq, StandardFont.child_equation_scale)
                if last_symbol_xmax is not None:
                    xshift = last_symbol_xmax
                else:
                    xshift = 0
                xrandom, yrandom = 0, 0
                yshift = 0
                if superscript_flag:
                    _, child_eq_h = get_equation_bbox_size(child_eq)
                    yshift = StandardFont.sup_equation_yshift - child_eq_h
                    superscript_flag = False
                elif subscript_flag:
                    yshift = StandardFont.sub_equation_yshift
                    subscript_flag = False
                child_eq = shift_equation(child_eq, xshift + xrandom, yshift + yrandom)
                last_symbol_xmax = get_equation_bbox(child_eq)[2]
                for trace_group in child_eq:
                    gen_equation.append(trace_group)
        return gen_equation


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
