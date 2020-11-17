import abc
import numpy as np


class RuleApplier(object):
    def __init__(self):
        self.rules = [
            CaseNormalizer(),
            One2Prime(),
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
        similar_letters = ['c', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        similar_letters = similar_letters + [letter.upper() for letter in similar_letters] + ['z' + letter.upper() for letter in similar_letters]
        self.similar_letters = similar_letters

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
            if node in self.uppercase and upper is False and node in self.similar_letters:
                return node.lstrip('z').lower()
            elif node in self.lowercase and upper is True and node in self.similar_letters:
                return node.upper()
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
        rate = summarize / np.sum(summarize) - self.threshold
        max_case = np.argmax(rate)
        max_rate = rate[max_case]
        min_count = np.min(summarize)
        if max_rate > 0 and min_count > 0:
            print('+ RULE APPLIED: "CASE NORMALIZER"')
            return self.normalize(lbst_tree, bool(max_case))
        return lbst_tree


class One2Prime(Rule):
    def normal_scan(self, node):
        if isinstance(node, list):
            for child in node:
                self.normal_scan(child)
            return node
        elif isinstance(node, dict):
            if 'type' in node and node['type'] == 'sup':
                self.normal_scan(node['child'][0])
                self.sup_scan(node['child'][1])
                return node
            if 'child' in node:
                for child in node['child']:
                    self.normal_scan(child)
            return node
        else:
            raise ValueError("type {} of {} not supported".format(type(node), node))

    def sup_scan(self, node):
        if isinstance(node, list):
            # for child in node:
            #     self.sup_scan(child)
            if len(node) > 0:
                self.sup_scan(node[0])
                for child in node[1:]:
                    self.normal_scan(child)
            return node
        elif isinstance(node, dict):
            if ('symbol' in node and node['symbol'] == '1') or ('type' in node and node['type'] == '1'):
                node['symbol'] = 'prime'
            if 'child' in node:
                for child in node['child']:
                    self.normal_scan(child)
            return node
        else:
            raise ValueError("type {} of {} not supported".format(type(node), node))

    def __call__(self, lbst_tree):
        return self.normal_scan(lbst_tree)
