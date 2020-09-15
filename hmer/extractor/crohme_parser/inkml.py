import os
import numpy as np

from xml.etree import ElementTree as ET

from constants import ink_xmlns, simplified_lbl_sep
from utilities import plt_draw
import re


class BaseTrait(object):
    """Base traits to be reuse"""
    _namespace = None
    _id = None

    @property
    def id(self):
        return self._id

    @property
    def namespace(self):
        return self._namespace


class Trace(BaseTrait):
    def __init__(self, element, namespace):
        """
        Represent a trace with id and coordinates
        :param element: an xml.etree.ElementTree.Element object represent trace information
        :param namespace:
        """
        self._namespace = namespace
        self._id = int(element.get('id'))
        self._coords = [
            [round(float(axis_coord) * 10000) for axis_coord in coord.strip().split(' ')]
            for coord in element.text.strip().split(',')]
        if len(self._coords[0]) == 2:
            pass
        elif len(self._coords[0]) == 3:
            # for data in CROHME2013 MfrDB dataset have coords with 3 elements
            # with the first element seems not very useful so we might as well eliminate it
            self._coords = [[coord[0], coord[1]] for coord in self._coords]

        self._bounding_box = [list(np.min(self._coords, axis=0)), list(np.max(self._coords, axis=0))]

    @property
    def coords(self):
        return self._coords

    @property
    def bbox(self):
        return self._bounding_box


class TraceGroup(BaseTrait):
    def __init__(self, element, namespace, traces_to_ref):
        """
        Represent a trace group with id, label and trace indices
        :param element: xml.etree.ElementTree.Element object represent trace group information
        :param namespace:
        :param traces_to_ref: List of traces for reference
        """
        self._namespace = namespace
        """
        :Note
        There are basically 2 (known) types of trace group id
        - a number. i.e. 1,2,3,....
        - list of number separate by colons. i.e. 0:, 1:2:3,...
        So there will be 2 ways of assigning id:
        - simply convert to int
        - generate a number id from 
        """
        _id = element.get([k for k in element.attrib.keys() if k.endswith('id')][0])
        self._id = TraceGroup._gen_id(_id, sep=':')

        self._label = element.find(namespace + "annotation").text

        #
        # Can't decide to split or merge frac and - together as the data does
        #
        # href = element.find(namespace + "annotationXML")
        #
        # if href is None:
        #     href = self._label
        # else:
        #     href = href.get('href')
        #
        # debug_p = re.compile(r'\d+:(\d+:)*')
        #
        # if self._label == '=' and not href.startswith('=') and not len(debug_p.findall(href)) > 0:
        #     print("DEBUG =")
        #
        # if self._label == '+' and not href.startswith('+') and not len(debug_p.findall(href)) > 0:
        #     print("DEBUG +")
        #
        # if self._label == '-' and (href.startswith('\\frac') or href.startswith('_')):
        #     self._label = '\\frac'
        # elif self._label == '-' and href.startswith('='):
        #     pass
        # elif len(debug_p.findall(href)) > 0:
        #     pass
        # elif self._label not in [
        #     '\\sqrt', '\\lt', '\\leq', '\\ldots', '\\gt', '\\geq', '.', '\\prime',
        #     '\\rightarrow', '\\neq', '\\exists', '\\sum', '\\int', '+'
        # ]:
        #     if not href.lstrip('\\').startswith(self._label.lstrip('\\')):
        #         print(f"WEIRD CASE: `{self._label}` and `{href}`")
        #         print("DEBUG")

        traces_idx = []
        for trace_view in element.findall(namespace + "traceView"):
            trace_data_ref = int(trace_view.get('traceDataRef'))
            traces_idx.append(trace_data_ref)

        self._traces = [traces_to_ref[idx] for idx in traces_idx]

        self._bounding_box = [coord for trace in self._traces for coord in trace.bbox]
        self._bounding_box = [list(np.min(self._bounding_box, axis=0)), list(np.max(self._bounding_box, axis=0))]

    @property
    def label(self):
        return self._label

    @property
    def traces(self):
        return self._traces

    @property
    def bbox(self):
        return self._bounding_box

    @staticmethod
    def _gen_id(id_lst, sep=':'):
        if sep not in id_lst:
            return int(id_lst)
        else:
            # id_lst = id_lst.strip(sep).split(sep)
            # id_scale = 10 ** int(max(map(lambda s: len(s), id_lst)))
            # return sum([(id_scale ** i) * int(_id) for i, _id in enumerate(id_lst)])
            return int(id_lst.replace(':', ''))

    @property
    def trace_coords(self):
        return [trace.coords for trace in self._traces]


class Ink(BaseTrait):
    def __init__(self, file_path, namespace=ink_xmlns, is_test=False):
        f"""
        Ink object contains all useful information extracted and to be used to visualize into image.
        :param file_path: inkml file path, absolute path works best
        :param namespace: namespace of the xml file. Default to be {ink_xmlns}
        """
        self._file_path = file_path
        self._namespace = namespace
        self._parsed = False
        self._tree = None
        self._traces = None
        self._trace_groups = None
        self._is_test = is_test

        self._parse_file()

    @property
    def file_path(self):
        return self._file_path

    @property
    def is_parsed(self):
        return self._parsed

    @property
    def label(self):
        return self._label

    @property
    def flatten_label(self):
        return ' '.join([g.label for g in self._trace_groups])

    @property
    def simplified_lbl(self):
        return self._simplified_label

    @property
    def traces(self):
        return self._traces

    @property
    def trace_groups(self):
        return self._trace_groups

    def _parse_file(self):
        """
        Parse file.
        Intended to be called from inside.
        :return:
        """
        if self._parsed:
            return None

        self._tree = tree = ET.parse(self._file_path)
        root = tree.getroot()

        if not self._is_test:
            lbl = [
                    child for child in root.getchildren()
                    if (child.tag == self._namespace + "annotation") and (child.attrib == {'type': 'truth'})
                ]
            if len(lbl) > 0:
                self._simplified_label = self._label = lbl[0].text
            else:
                print("ERROR IN FILE: {}.\nRetry with finding typx: \"truth\" instead.".format(self._file_path))
                lbl = [
                    child for child in root.getchildren()
                    if (child.tag == self._namespace + "annotation") and (child.attrib == {'typx': 'truth'})
                ]
                self._simplified_label = self._label = lbl
        else:
            self._simplified_label = self._label = None

        self._traces = traces = [Trace(trace_tag, self._namespace) for trace_tag in
                                 root.findall(self._namespace + "trace")]
        # since trace always start with index 0, we can easily index from array, not necessary to turn it into dict
        traces.sort(key=lambda trace: trace.id)

        trace_group_wrapper = root.find(self._namespace + "traceGroup")

        if trace_group_wrapper is not None:
            trace_groups = [
                TraceGroup(trace_group_element, self._namespace, self._traces)
                for trace_group_element in trace_group_wrapper.findall(self._namespace + "traceGroup")]

            trace_groups.sort(key=lambda group: group.bbox[0][0])

            self._trace_groups = trace_groups

            self._simplified_label = simplified_lbl_sep.join([
                group.label
                for group in trace_groups
            ])

        self._parsed = True

    @property
    def trace_coords(self):
        if not self._is_test:
            return [trace_coords for group in self._trace_groups for trace_coords in group.trace_coords]
        else:
            return [trace.coords for trace in self._traces]

    def convert_to_img(self, output_path, write_simplified_label=False, linewidth=2, color='b', draw_bbox=False,
                       **draw_bbox_kwargs):
        """
        Convert Ink object to image using matplotlib
        :param output_path: path to write image and label. please provide path without suffix.
        :param write_simplified_label:
        :param linewidth:
        :param color:
        :param draw_bbox: whether to plot bbox or not
        :return:
        """
        assert self._parsed, "Please parse the file before furthur operation!"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt_draw.plt_clear()
        # write label
        with open(output_path + '.label.txt', 'w+') as fout:
            if write_simplified_label:
                fout.write(self._simplified_label)
            else:
                fout.write(self._label)

        fig, ax = plt_draw.plt_setup()

        if self._trace_groups is not None:
            for group in self._trace_groups:
                for trace in group.traces:
                    plt_draw.plt_trace_coords(trace.coords, ax=ax, linewidth=linewidth, c=color)
                if draw_bbox:
                    plt_draw.plt_draw_bbox(group.bbox, ax=ax, **draw_bbox_kwargs)
        else:
            for trace in self._traces:
                plt_draw.plt_trace_coords(trace.coords, ax=ax, linewidth=linewidth, c=color)
                if draw_bbox:
                    plt_draw.plt_draw_bbox(trace.bbox, ax=ax, **draw_bbox_kwargs)
        plt_draw.plt_savefig(output_path + '.png')


def parse_inkml_dir(data_dir_abs_path):
    if os.path.isdir(data_dir_abs_path):
        for inkml_file in os.listdir(data_dir_abs_path):
            if inkml_file.endswith('.inkml'):
                inkml_file_abs_path = os.path.join(data_dir_abs_path, inkml_file)
                print("Parsing: {}".format(inkml_file_abs_path))
                yield Ink(inkml_file_abs_path)


if __name__ == '__main__':
    ink = Ink("../../data/CROHME_full_v2/CROHME2013_data/TrainINKML/HAMEX/formulaire001-equation003.inkml")
    ink.convert_to_img("test", write_simplified_label=True)
