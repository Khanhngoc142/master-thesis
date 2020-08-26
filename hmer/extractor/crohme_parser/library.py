import os
import random
import warnings

from extractor.crohme_parser.inkml import Ink
from extractor.crohme_parser.extract import Extractor
from utils.fs import get_source_root
from utils.image_processing import get_trace_group_bbox, shift_trace_group_coords, centerize_trace_group_coords, \
    scale_trace_group, scale2height_trace_group, scale2width_trace_group, scale_trace_group_v2, scale_equation, \
    get_equation_bbox_size, shift_equation, get_equation_bbox, get_trace_group_bbox_size
from extractor.crohme_parser.fonts import StandardFont
from enum import Enum, auto


class EquationFlag(Enum):
    SUPSCRIPT = auto()
    SUBSCRIPT = auto()
    LIM = auto()
    LIMLOWER = auto()
    SUM = auto()
    SUMUPPER = auto()
    SUMLOWER = auto()
    INTEGRAL = auto()
    INTEGRALUPPER = auto()
    INTEGRALLOWER = auto()
    SQRT = auto()


def get_scaledown_flag():
    return [
        EquationFlag.SUPSCRIPT,
        EquationFlag.SUBSCRIPT,
        EquationFlag.LIMLOWER,
        EquationFlag.SUMUPPER,
        EquationFlag.SUMLOWER,
    ]


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
        last_symbol_bbox = None
        flag = None
        for e in equation:
            # handle symbols
            if isinstance(e, str):
                if e == '^':
                    if flag is None:
                        flag = EquationFlag.SUPSCRIPT
                    elif flag is EquationFlag.SUM:
                        flag = EquationFlag.SUMUPPER
                    elif flag is EquationFlag.INTEGRAL:
                        flag = EquationFlag.INTEGRALUPPER
                    continue
                elif e == '_':
                    if flag is None:
                        flag = EquationFlag.SUBSCRIPT
                    elif flag is EquationFlag.LIM:
                        flag = EquationFlag.LIMLOWER
                    elif flag is EquationFlag.SUM:
                        flag = EquationFlag.SUMLOWER
                    elif flag is EquationFlag.INTEGRAL:
                        flag = EquationFlag.INTEGRALLOWER
                    continue

                _, cur_symbol = self.get_sample_of_symbol(e)
                cur_scale, cur_yshift = StandardFont.font_metric[e]

                # there are different scale tragetries depending on which characters
                # scale by width
                if e in list('-+=') + ['\\infty', '\\ldots', '\\pm']:
                    _, cur_symbol = scale2width_trace_group(cur_symbol, cur_scale)
                # scale by max side
                elif e in list(',.') + [
                    '\\rightarrow', '\\times', '\\div',
                    '\\sum', '\\exists', '\\forall',
                    '\\geq', '\\gt', '\\leq', '\\lt', '\\neq',
                ]:
                    _, cur_symbol = scale_trace_group_v2(cur_symbol, cur_scale)
                # scale by height. Default
                else:
                    _, cur_symbol = scale2height_trace_group(cur_symbol, cur_scale)

                xrandom, yrandom = 0, 0  # TODO: implement some random shift

                # xshift
                if last_symbol_bbox is not None:
                    xshift = last_symbol_bbox[2] + StandardFont.symbol_gap
                else:
                    xshift = 0
                # special case shift y to center at cur_yshift
                _, h = get_trace_group_bbox_size(cur_symbol)
                if e in list('+-=') + [
                    '\\rightarrow', '\\times', '\\div',
                    '\\geq', '\\gt', '\\leq', '\\lt', '\\neq',
                    '\\infty',
                    '\\ldots',
                ]:
                    cur_yshift -= h / 2
                # yshift bottom alignment
                elif e in ['\\pm']:
                    cur_yshift -= h
                elif e == '\\prime':
                    cur_yshift += last_symbol_bbox[1]

                # shift
                cur_symbol = shift_trace_group_coords(cur_symbol, xshift=xshift + xrandom, yshift=cur_yshift + yrandom)

                # update info var
                last_symbol_bbox = get_trace_group_bbox(cur_symbol)
                gen_equation.append(cur_symbol)

                # reset flag
                if e == '\\lim':
                    flag = EquationFlag.LIM
                elif e == '\\sum':
                    flag = EquationFlag.SUM
                elif e == '\\int':
                    flag = EquationFlag.INTEGRAL
                elif e == '\\sqrt':
                    flag = EquationFlag.SQRT
                else:
                    flag = None
            # handle child equations
            elif isinstance(e, list):
                child_eq = self.generate_equation_traces(e)

                # scale child equation
                if flag in [EquationFlag.SUPSCRIPT, EquationFlag.SUBSCRIPT, ]:
                    child_eq = scale_equation(child_eq, StandardFont.supsub_equation_scale)
                elif flag in [
                    EquationFlag.LIMLOWER,
                    EquationFlag.SUMUPPER, EquationFlag.SUMLOWER,
                    EquationFlag.INTEGRALLOWER, EquationFlag.INTEGRALUPPER
                ]:
                    child_eq = scale_equation(child_eq, StandardFont.lowerupper_equation_scale)
                elif flag in [
                    EquationFlag.SQRT
                ]:
                    child_eq = scale_equation(child_eq, StandardFont.sqrt_equation_scale)

                # x shift
                # center child symbol to center of last symbol, last_symbol_bbox is always not None in THIS case
                if flag in [
                    EquationFlag.LIMLOWER,
                    EquationFlag.SUMLOWER, EquationFlag.SUMUPPER
                ]:
                    child_xmin, child_ymin, child_xmax, child_ymax = get_equation_bbox(child_eq)
                    child_w = child_xmax - child_xmin
                    last_symbol_w = last_symbol_bbox[2] - last_symbol_bbox[0]
                    # upperlower center
                    if flag in [EquationFlag.SUMUPPER, ]:
                        xshift = last_symbol_bbox[0] - StandardFont.upperlower_xshift_center(child_w, last_symbol_w)
                    # upperlower leftalign. Default
                    else:
                        xshift = last_symbol_bbox[0] - StandardFont.upperlower_xshift_leftalign(child_w, last_symbol_w)
                elif flag is EquationFlag.INTEGRALLOWER:
                    xshift = last_symbol_bbox[2] - (last_symbol_bbox[2] - last_symbol_bbox[0]) / 3
                elif last_symbol_bbox is not None:
                    if flag is EquationFlag.SQRT:
                        if e[0] == "\\sqrt":
                            xshift = last_symbol_bbox[0] + 3 * StandardFont.symbol_gap
                        else:
                            xshift = last_symbol_bbox[0] + StandardFont.symbol_gap
                    else:
                        xshift = \
                            last_symbol_bbox[2] \
                            + StandardFont.supsub_equation_gap_from_parent
                else:
                    xshift = 0

                # yshift
                yshift = 0
                _, child_eq_h = get_equation_bbox_size(child_eq)
                # simple superscript
                if flag is EquationFlag.SUPSCRIPT:
                    yshift = StandardFont.sup_equation_yshift - child_eq_h
                # simple subscript
                elif flag is EquationFlag.SUBSCRIPT:
                    yshift = StandardFont.sub_equation_yshift
                elif flag in [EquationFlag.LIMLOWER, EquationFlag.SUMLOWER]:
                    yshift = last_symbol_bbox[3] + StandardFont.lower_equation_yshift
                elif flag is EquationFlag.SUMUPPER:
                    yshift = last_symbol_bbox[1] - StandardFont.upper_equation_yshift - child_eq_h
                elif flag is EquationFlag.INTEGRALUPPER:
                    yshift = last_symbol_bbox[1] - child_eq_h / 2
                elif flag is EquationFlag.INTEGRALLOWER:
                    yshift = last_symbol_bbox[3] - child_eq_h / 2
                elif flag is EquationFlag.SQRT:
                    _, _, _, child_ymax = get_equation_bbox(child_eq)
                    yshift = 1 - child_ymax
                    # if e[0] == "\\sqrt":
                    #     yshift = last_symbol_bbox[3] - 0.9 * child_eq_h
                    # else:
                    #     yshift = last_symbol_bbox[3] - child_eq_h
                    # yshift = last_symbol_bbox[3] - child_eq_h + 0.5
                    # yshift = 0

                # distabilize position
                xrandom, yrandom = 0, 0

                # shift equation
                child_eq = shift_equation(child_eq, xshift + xrandom, yshift + yrandom)

                # update info var
                # in some special cases we maintain the last_symbol_bbox
                if flag in [
                    EquationFlag.LIMLOWER,
                    EquationFlag.SUMUPPER, EquationFlag.SUMLOWER,
                    EquationFlag.SQRT,
                ]:
                    last_symbol_bbox = [min(*child_parent_bbox_pair) if i < 2 else max(*child_parent_bbox_pair) for
                                        i, child_parent_bbox_pair in
                                        enumerate(zip(get_equation_bbox(child_eq), last_symbol_bbox))]
                elif flag in [
                    EquationFlag.INTEGRALUPPER, EquationFlag.INTEGRALLOWER,
                ]:
                    pass
                else:
                    last_symbol_bbox = get_equation_bbox(child_eq)

                for trace_group in child_eq:
                    gen_equation.append(trace_group)

                # reset flag
                if flag in [EquationFlag.SUMUPPER, EquationFlag.SUMLOWER]:
                    flag = EquationFlag.SUM
                elif flag in [EquationFlag.INTEGRALUPPER, EquationFlag.INTEGRALLOWER]:
                    flag = EquationFlag.INTEGRAL
                else:
                    flag = None
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
