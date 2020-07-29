import os
import numpy as np
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

from constants import ink_xmlns


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

    @property
    def coords(self):
        return self._coords


class TraceGroup(BaseTrait):
    def __init__(self, element, namespace):
        """
        Represent a trace group with id, label and trace indices
        :param element: xml.etree.ElementTree.Element object represent trace group information
        :param namespace:
        """
        self._namespace = namespace
        self._id = int(element.get([k for k in element.attrib.keys() if k.endswith('id')][0]))
        self._label = element.find(namespace + "annotation").text

        self._traces_idx = traces_idx = []
        for trace_view in element.findall(namespace + "traceView"):
            trace_data_ref = int(trace_view.get('traceDataRef'))
            traces_idx.append(trace_data_ref)

    @property
    def label(self):
        return self._label

    @property
    def traces_idx(self):
        return

    def get_traces(self, traces_to_ref):
        for idx in self._traces_idx:
            yield traces_to_ref[idx]


class Ink(BaseTrait):
    def __init__(self, file_path, namespace=ink_xmlns):
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
    def simplified_label(self):
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

        self._simplified_label = self._label = [
            child for child in root.getchildren()
            if (child.tag == self._namespace + "annotation") and (child.attrib == {'type': 'truth'})
        ][0].text

        self._traces = traces = [Trace(trace_tag, self._namespace) for trace_tag in
                                 root.findall(self._namespace + "trace")]
        # since trace always start with index 0, we can easily index from array, not necessary to turn it into dict
        traces.sort(key=lambda trace: trace.id)

        trace_group_wrapper = root.find(self._namespace + "traceGroup")

        if trace_group_wrapper is not None:
            trace_groups = [
                TraceGroup(trace_group_element, self._namespace)
                for trace_group_element in trace_group_wrapper.findall(self._namespace + "traceGroup")]

            trace_groups.sort(key=lambda group: group.id)
            self._trace_groups = trace_groups

            self._simplified_label = ''.join([
                group.label
                for group in trace_groups
            ])

        self._parsed = True

    def convert_to_img(self, output_path, write_simplified_label=False):
        """
        Convert Ink object to image
        :param write_simplified_label:
        :param output_path: path to write image and label. please provide path without suffix.
        :return:
        """
        assert self._parsed, "Please parse the file before furthur operation!"
        # write label
        with open(output_path + '.label.txt', 'w+') as fout:
            if write_simplified_label:
                fout.write(self._simplified_label)
            else:
                fout.write(self._label)

        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.axes().spines['top'].set_visible(False)
        plt.axes().spines['right'].set_visible(False)
        plt.axes().spines['bottom'].set_visible(False)
        plt.axes().spines['left'].set_visible(False)

        if self._trace_groups is not None:
            for group in self._trace_groups:
                traces = group.get_traces(self._traces)
                for trace in traces:
                    data = np.array(trace.coords)
                    x, y = zip(*data)
                    plt.plot(x, y, linewidth=2, c='black')
        else:
            for trace in self._traces:
                data = np.array(trace.coords)
                x, y = zip(*data)
                plt.plot(x, y, linewidth=2, c='black')
        plt.savefig(output_path + '.png', bbox_inches='tight', dpi=100)
        plt.gcf().clear()


if __name__ == '__main__':
    ink = Ink("/home/lap13639/Workplace/git/github/master-thesis/hmer/data/CROHME_full_v2/CROHME2013_data/TrainINKML/HAMEX/formulaire001-equation002.inkml")
    ink.convert_to_img("test")
