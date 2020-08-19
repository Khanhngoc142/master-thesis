import os
import random
import warnings

from extractor.crohme_parser.inkml import Ink
from extractor.crohme_parser.extract import Extractor
from utils.fs import get_source_root
from utils.image_processing import get_trace_group_bbox, shift_trace_group_coords, centerize_trace_group_coords, scale_trace_group


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
        self._font_metric = {  # font metric: symbols -> (standard_scale, standard_y_shift)
            '!': (1, 0),
            '(': (1.1, 0),
            ')': (1.1, 0),
            '+': (0.6, 0.3),
            ',': (0.3, 0.8),
            '-': (0.6, 0.3),
            '.': (0.3, 0.6),
            '/': (1, 0),
            '0': (1, 0),
            '1': (1, 0),
            '2': (1, 0),
            '3': (1, 0),
            '4': (1, 0),
            '5': (1, 0),
            '6': (1, 0),
            '7': (1, 0),
            '8': (1, 0),
            '9': (1, 0),
            '=': (0.6, 0.3),
            'A': (1, 0),
            'B': (1, 0),
            'C': (1, 0),
            'E': (1, 0),
            'F': (1, 0),
            'G': (1, 0),
            'H': (1, 0),
            'I': (1, 0),
            'L': (1, 0),
            'M': (1, 0),
            'N': (1, 0),
            'P': (1, 0),
            'R': (1, 0),
            'S': (1, 0),
            'T': (1, 0),
            'V': (1, 0),
            'X': (1, 0),
            'Y': (1, 0),
            '[': (1.1, 0),
            # '\\Delta': (,),
            # '\\alpha': (,),
            # '\\beta': (,),
            # '\\cos': (,),
            # '\\div': (,),
            # '\\exists': (,),
            # '\\forall': (,),
            # '\\gamma': (,),
            # '\\geq': (,),
            # '\\gt': (,),
            # '\\in': (,),
            # '\\infty': (,),
            # '\\int': (,),
            # '\\lambda': (,),
            # '\\ldots': (,),
            # '\\leq': (,),
            # '\\lim': (,),
            # '\\log': (,),
            # '\\lt': (,),
            # '\\mu': (,),
            # '\\neq': (,),
            # '\\phi': (,),
            # '\\pi': (,),
            # '\\pm': (,),
            # '\\prime': (,),
            # '\\rightarrow': (,),
            # '\\sigma': (,),
            # '\\sin': (,),
            # '\\sqrt': (,),
            # '\\sum': (,),
            # '\\tan': (,),
            # '\\theta': (,),
            # '\\times': (,),
            '\\{': (1, 0),
            '\\}': (1, 0),
            ']': (1.1, 0),
            'a': (0.8, 0.2),
            'b': (1, 0),
            'c': (0.8, 0.2),
            'd': (1, 0),
            'e': (0.8, 0.2),
            'f': (1.2, 0),
            'g': (1.1, 0.5),
            'h': (1, 0),
            'i': (0.8, 0.2),
            'j': (1.2, 0.5),
            'k': (1, 0),
            'l': (1, 0),
            'm': (1, 0.4),
            'n': (0.8, 0.4),
            'o': (0.8, 0.2),
            'p': (1.1, 0.6),
            'q': (1.1, 0.6),
            'r': (0.8, 0.2),
            's': (0.8, 0.2),
            't': (0.9, 0.1),
            'u': (0.8, 0.2),
            'v': (0.8, 0.2),
            'w': (0.8, 0.2),
            'x': (0.8, 0.2),
            'y': (1.1, 0.6),
            'z': (0.8, 0.2),
            '|': (1.1, 0),
        }

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

    def generate_equation_traces(self, equation, scale=1):
        """
        Generate euation traces from random sample
        :param scale:
        :param equation:
        :return: list of list of traces
        """
        gen_equation = []
        last_symbol_bbox = None
        for e in equation:
            if isinstance(e, str):
                _, cur_symbol = self.get_sample_of_symbol(e)
                cur_scale, cur_yshift = self._font_metric[e]
                if cur_scale != 1:
                    _, cur_symbol = scale_trace_group(cur_symbol, cur_scale)
                if last_symbol_bbox is not None:
                    xmax = last_symbol_bbox[2]
                    # ymax = last_symbol_bbox[3]  # TODO: do something with this?
                    xrandom, yrandom = 0.2, 0  # TODO: implement some random shift
                    cur_symbol = shift_trace_group_coords(cur_symbol, xshift=xmax + xrandom, yshift=cur_yshift + yrandom)
                last_symbol_bbox = get_trace_group_bbox(cur_symbol)
                gen_equation.append(cur_symbol)
            else:
                pass

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
