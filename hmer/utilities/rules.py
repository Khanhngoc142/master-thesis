import abc
import numpy as np


class RuleApplier(object):
    def __init__(self):
        self.rules = [
            CaseNormalizer(),
        ]

    def __call__(self, lbst_tree):
        assert isinstance(lbst_tree, list), "input must be a region (list of nodes)"
        for rule in self.rules:
            lbst_tree = rule(lbst_tree)
        return lbst_tree


class Rule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, lbst_tree):
        pass


class CaseNormalizer(Rule):
    def __init__(self, threshold=0.5):
        self.lowercase = [chr(c) for c in range(ord('a'), ord('z') + 1)]
        self.uppercase = ['z' + c.upper() for c in self.lowercase] + [c.upper() for c in self.lowercase]
        self.threshold = threshold

    def trace_symbol(self, node):
        if isinstance(node, str):
            if node in self.uppercase:
                return np.array([0, 1])
            elif node in self.lowercase:
                return np.array([1, 0])
            else:
                return np.array([0, 0])
        base = np.array([0, 0])
        if isinstance(node, dict):
            if 'symbol' in node:
                base += self.trace_symbol(node['symbol'])
            if 'child' in node:
                for child in node['child']:
                    base += self.trace_symbol(child)
        elif isinstance(node, list):
            for child in node:
                base += self.trace_symbol(child)
        else:
            raise ValueError("node type {} not supported".format(node))

        return base

    def normalize(self, node, upper):
        if isinstance(node, str):
            if node in self.uppercase and upper == False:
                return node.lstrip('z').lower()
            elif node in self.lowercase and upper == True:
                return 'z' + node.upper()
            else:
                return node
        if isinstance(node, dict):
            if 'symbol' in node:
                node['symbol'] = self.normalize(node['symbol'], upper)
            if 'child' in node:
                for child in node['child']:
                    _ = self.normalize(child, upper)
        elif isinstance(node, list):
            for child in node:
                _ = self.normalize(child, upper)
        return node

    def __call__(self, lbst_tree):
        # summarize
        summarize = self.trace_symbol(lbst_tree)
        summarize = summarize / np.sum(summarize) - self.threshold
        max_case = np.argmax(summarize)
        max_rate = summarize[max_case]
        if max_rate > 0:
            print('+ RULE APPLIED: "CASE NORMALIZER"')
            return self.normalize(lbst_tree, bool(max_case))
        return lbst_tree
