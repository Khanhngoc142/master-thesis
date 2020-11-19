import abc
import numpy as np


class RuleApplier(object):
    def __init__(self):
        self.rules = [
            Time2X(),
            CaseNormalizer(),
            One2Prime(),
            O2Zero(),
            OneCorrecterTwoSides(),
            OneCorrecterAfter(),
        ]

    def __call__(self, lbst_tree):
        assert isinstance(lbst_tree, list), "input must be a region (list of nodes)"
        for rule in self.rules:
            lbst_tree = rule(lbst_tree)
        return lbst_tree


class Rule(metaclass=abc.ABCMeta):
    def __init__(self):
        self.verbose_pattern = '+ RULE APPLIED: "{}"'
        self.rule_name = ""

    def verbose(self):
        print(self.verbose_pattern.format(self.rule_name))

    def type_not_supported(self, node):
        raise ValueError("DEBUG: {}: type {} not supported".format(self.rule_name, type(node)))

    @abc.abstractmethod
    def __call__(self, lbst_tree):
        pass


class CaseNormalizer(Rule):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.rule_name = "CASE NORMALIZE"
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
            if 'type' in node:
                base += self.trace_symbol(node['type'])
        elif isinstance(node, list):
            for child in node:
                base += self.trace_symbol(child)
        else:
            self.type_not_supported(node)

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
            if 'type' in node:
                node['type'] = self.normalize(node['type'], upper)
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
        sum_summarize = np.sum(summarize)
        if sum_summarize == 0:
            return lbst_tree
        rate = summarize / sum_summarize - self.threshold
        max_case = np.argmax(rate)
        max_rate = rate[max_case]
        min_count = np.min(summarize)
        if max_rate > 0 and min_count > 0:
            self.verbose()
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
            self.type_not_supported(node)

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
                self.verbose()
                node['symbol'] = 'prime'
            if 'type' in node and node['type'] == '1':
                self.verbose()
                node['type'] = 'prime'
            if 'child' in node:
                for child in node['child']:
                    self.normal_scan(child)
            return node
        else:
            self.type_not_supported(node)

    def __call__(self, lbst_tree):
        return self.normal_scan(lbst_tree)


class O2Zero(Rule):
    def __init__(self):
        super().__init__()
        self.rule_name = "o TO 0"

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
            self.type_not_supported(node)

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
                self.verbose()
                node['symbol'] = '0'
            if 'type' in node and node['type'] in ['o', 'O']:
                self.verbose()
                node['type'] = '0'
            if 'child' in node:
                for child in node['child']:
                    self.normal_scan(child)
            return node
        else:
            raise ValueError("type {} of {} not supported".format(type(node), node))

    def __call__(self, lbst_tree):
        return self.normal_scan(lbst_tree)


class OneCorrecterTwoSides(Rule):
    """
    if there's any | next to (before or after) an op => | to 1
    """
    def __init__(self):
        super().__init__()
        self.target_symbols = ['|']
        self.rule_name = '1 CORRECTOR 2 SIDES'

    def modify(self, symbol):
        if isinstance(symbol, dict):
            if 'type' in symbol and symbol['type'] in ['sup', 'sub']:
                self.modify(symbol['child'][0])
            elif 'symbol' in symbol and symbol['symbol'].strip() in self.target_symbols:
                self.verbose()
                symbol['symbol'] = '1'
                symbol['type'] = 'literal'
            elif 'type' in symbol and symbol['type'].strip() in self.target_symbols:
                self.verbose()
                symbol['type'] = '1'
        elif isinstance(symbol, list):
            if len(symbol) > 0:
                self.modify(symbol[0])
        else:
            self.type_not_supported(symbol)

    def __call__(self, lbst_tree):
        if not isinstance(lbst_tree, list):
            self.type_not_supported(lbst_tree)

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
                self.type_not_supported(curr_symbol)
            i += 1
        return lbst_tree


class OneCorrecterAfter(OneCorrecterTwoSides):
    """
    if there's any | next to (before or after) an op => | to 1
    """
    def __init__(self):
        super().__init__()
        self.target_symbols = ['!', 'slash', '/']
        self.rule_name = '1 CORRECTOR AFTER'

    def __call__(self, lbst_tree):
        if not isinstance(lbst_tree, list):
            self.type_not_supported(lbst_tree)

        i = 0
        while i < len(lbst_tree):
            curr_symbol = lbst_tree[i]
            if isinstance(curr_symbol, list):
                self(curr_symbol)
            elif isinstance(curr_symbol, dict):
                if 'type' in curr_symbol and curr_symbol['type'] == 'Operation':
                    if i + 1 < len(lbst_tree):
                        self.modify(lbst_tree[i + 1])
                if 'child' in curr_symbol:
                    for child in curr_symbol['child']:
                        self(child)
            else:
                self.type_not_supported(curr_symbol)
            i += 1
        return lbst_tree


class Time2X(Rule):
    def __init__(self):
        super().__init__()
        self.rule_name = "TIME TO X"
        self.target_symbols = ['time', '\\times', 'times', '\\time']
        self.out_symbol = 'x'
        # self.out_symbol = 'zX'

    def modify(self, node):
        assert isinstance(node, dict)
        if 'type' in node and node['type'] in self.target_symbols:
            self.verbose()
            node['type'] = self.out_symbol
        if 'type' in node and node['type'] == 'Operation' and 'symbol' in node and node['symbol'] in self.target_symbols:
            self.verbose()
            node['symbol'] = self.out_symbol
            node['type'] = 'literal'

    def __call__(self, lbst_tree):
        if isinstance(lbst_tree, list):
            if len(lbst_tree) > 0:
                if isinstance(lbst_tree[0], list):
                    for child in lbst_tree:
                        self(child)
                elif isinstance(lbst_tree[0], dict):
                    if len(lbst_tree) > 0:
                        for i in [0, -1]:
                            self.modify(lbst_tree[i])
                    for i, child in enumerate(lbst_tree[1:-1]):
                        if 'type' in child and child['type'] in self.target_symbols:
                            self.verbose()
                            child['type'] = self.out_symbol
                        if 'type' in child and child['type'] == 'Operation' and child['symbol'] in self.target_symbols:
                            pre_symbol = lbst_tree[i]
                            if 'type' in pre_symbol and pre_symbol['type'] == 'Operation':
                                self.verbose()
                                child['symbol'] = self.out_symbol
                                child['type'] = 'literal'
                else:
                    self.type_not_supported(lbst_tree[0])
        else:
            self.type_not_supported(lbst_tree)

        return lbst_tree
