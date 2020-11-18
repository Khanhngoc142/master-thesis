import abc
import numpy as np


class RuleApplier(object):
    def __init__(self):
        self.rules = [
            CaseNormalizer(),
            One2Prime(),
            O2Zero(),
            OneCorrecter(),
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
        similar_letters = ['c', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z']
        similar_letters = similar_letters + [letter.upper() for letter in similar_letters] + ['z' + letter.upper() for letter in similar_letters]
        self.similar_letters = similar_letters
        # self.similar_letters = []

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
            if 'symbol' in node and node['symbol'] == '1':
                print('+ RULE APPLIED: "1 TO prime"')
                node['symbol'] = 'prime'
            if 'type' in node and node['type'] == '1':
                print('+ RULE APPLIED: "1 TO prime"')
                node['type'] = 'prime'
            if 'child' in node:
                for child in node['child']:
                    self.normal_scan(child)
            return node
        else:
            raise ValueError("type {} of {} not supported".format(type(node), node))

    def __call__(self, lbst_tree):
        return self.normal_scan(lbst_tree)


class O2Zero(Rule):
    def normal_scan(self, node):
        if isinstance(node, list):
            for child in node:
                self.normal_scan(child)
            return node
        elif isinstance(node, dict):
            if 'type' in node and node['type'] == 'sub':
                self.normal_scan(node['child'][0])
                self.sub_scan(node['child'][1])
                return node
            if 'child' in node:
                for child in node['child']:
                    self.normal_scan(child)
            return node
        else:
            raise ValueError("type {} of {} not supported".format(type(node), node))

    def sub_scan(self, node):
        if isinstance(node, list):
            # for child in node:
            #     self.sup_scan(child)
            if len(node) > 0:
                self.sub_scan(node[0])
                for child in node[1:]:
                    self.normal_scan(child)
            return node
        elif isinstance(node, dict):
            if 'symbol' in node and node['symbol'] in ['o', 'O']:
                print('+ RULE APPLIED: "o TO 0"')
                node['symbol'] = '0'
            if 'type' in node and node['type'] in ['o', 'O']:
                print('+ RULE APPLIED: "o TO 0"')
                node['type'] = '0'
            if 'child' in node:
                for child in node['child']:
                    self.normal_scan(child)
            return node
        else:
            raise ValueError("type {} of {} not supported".format(type(node), node))

    def __call__(self, lbst_tree):
        return self.normal_scan(lbst_tree)


class OneCorrecter(Rule):
    """
    if there's any | next to (before or after) an op => | to 1
    """
    def modify(self, symbol):
        if isinstance(symbol, dict):
            if 'type' in symbol and symbol['type'] in ['sup', 'sub']:
                self.modify(symbol['child'][0])
            elif 'symbol' in symbol and symbol['symbol'] == '|':
                symbol['symbol'] = '1'
            elif 'type' in symbol and symbol['type'] == '|':
                symbol['type'] = '1'
        elif isinstance(symbol, list):
            if len(symbol) > 0:
                self.modify(symbol[0])
        else:
            raise ValueError("DEBUG: OneCorrector: type {} not supported".format(type(symbol)))

    def __call__(self, lbst_tree):
        if not isinstance(lbst_tree, list):
            raise ValueError("DEBUG: OneCorrector: type {} not supported".format(type(lbst_tree)))

        i = 0
        while i < len(lbst_tree):
            curr_symbol = lbst_tree[i]
            if isinstance(curr_symbol, list):
                self(curr_symbol)
            elif isinstance(curr_symbol, dict):
                if 'type' in curr_symbol and curr_symbol['type'] == 'Operation':
                    if i + 1 < len(lbst_tree):
                        self.modify(lbst_tree[i + 1])
                    if i - 1 >= 0:
                        self.modify([lbst_tree[i - 1]])
                if 'child' in curr_symbol:
                    for child in curr_symbol['child']:
                        self(child)
            else:
                raise ValueError("DEBUG: OneCorrector: type {} not supported".format(type(curr_symbol)))
            i += 1
        return lbst_tree
