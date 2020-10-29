from pprint import pprint
from operator import itemgetter
import string
import copy
from PIL import Image, ImageDraw, ImageFont

# centroid_ratio = 1/2
# threshold_ratio = 3/4


def and_all(lst):
    if len(lst) == 1:
        return lst[0]
    return lst[0] and and_all(lst[1:])


def is_the_same(snode_1, snode_2):
    return and_all([snode_1[i] == snode_2[i] for i in range(5)])


class SymbolManager:
    def __init__(self):

        # print('initializing symbols ...')

        # structure of an entry:
        # (sym, index, class, Alignment)
        self.dictionary = []

        with open('./new_label.txt') as f:
            sym_list = f.readlines()
            for sym in sym_list:
                temp = sym.strip().split()
                self.dictionary.append([temp[0], int(temp[1])])

        self.add_class_to_dictionary()

    def add_class_to_dictionary(self):
        # Class consist of
        #  NonScripted: + - = > ->
        non_scripted = ['add', 'sub', '=', 'rightarrow', 'leq', 'geq', 'neq', 'lt', 'gt']  # 7
        #  Bracket ( { [
        bracket = ['(', '[', '\\{']
        close_brackets = [')', ']', '\\}']
        #  Root : sqrt
        square_root = ['sqrt']
        #  VariableRange : _Sigma, integral, _PI,
        variable_range = ['_Sigma', '_Pi', 'lim', 'integral']
        #  Plain_Ascender: 0..9, A..Z, b d f h i k l t
        plain_ascender = ['b', 'd', 'f', 'h', 'i', 'k', 'l', 't', 'exists', 'forall', '!', '_Delta', '_Omega', '_Phi', 'div', 'beta', 'lamda', 'tan', 'log', '|']
        plain_ascender = plain_ascender + list(map(str, range(10)))
        plain_ascender = plain_ascender + list(string.ascii_uppercase)
        #  Plain_Descender: g p q y gamma, nuy, rho khi phi
        plain_decender = ['g', 'p', 'q', 'y', 'gamma', 'muy', 'rho', '']
        #  plain_Centered: The rest
        for entry in self.dictionary:
            temp_class = 'plain_Centered'
            alignment = 'Centred'
            if entry[0] in non_scripted:
                temp_class = 'NonScripted'
            elif entry[0] in bracket:
                temp_class = 'Bracket'
                alignment = 'Ascender'
            elif entry[0] in square_root:
                temp_class = 'Root'
                alignment = 'Ascender'
            elif entry[0] in variable_range:
                temp_class = 'VariableRange'
            elif entry[0] in plain_ascender:
                temp_class = 'Plain_Ascender'
                alignment = 'Ascender'
            elif entry[0] in plain_decender:
                temp_class = 'Plain_Descender'
                alignment = 'Descender'
            elif entry[0] in close_brackets:
                temp_class = 'CloseBracket'
            entry.append(temp_class)
            entry.append(alignment)

    def get_class(self, symbol):
        for entry in self.dictionary:
            if entry[0] == symbol:
                return entry[2]

    def get_symbol_from_idx(self, index):
        for entry in self.dictionary:
            if entry[1] == index:
                return entry[0], entry[2], entry[3]


class LBST:
    def __init__(self, psymbol_manager):
        # root = []
        self.symbol_manager = psymbol_manager

        self.compoundable_list = list('0123456789abcdefghijklmnopqrstuvwz') + ['ldot']
        self.Prefix_compoundable_list = list('ABCDEFGHIJKLMNOPQRSTUVWZ')
        self.OperatorList_eq = ['=', 'neq', 'in', 'geq', 'leq']
        self.OperatorList_as = ['add', 'sub']
        self.OperatorList_md = ['time', 'div', 'slash']

        self.AllOp = self.OperatorList_eq + self.OperatorList_as + self.OperatorList_md

        self.AllowAjacentAsMultiply = ['sub', 'sup', 'literal', 'sqrt']  # please handle f(x)

        self.AllowAjacentAsMultiplyRight = ['sin', 'cos', 'tan', 'log', 'lim']

        self.VariableRangeList = ['_Sigma', '_Pi', 'integral', 'lim', 'frac', 'rightarrow']

    # RULE TO PARSE :
    #:
    # SUM
    # PI
    #     sub sup
    # frac
    # int
    #     sqrt
    # lim
    # rightarrow

    # child_temp: TL, BL, T, B, C, SUP, SUB

    ############################
    # LBSTnode_sym
    # sym, BST_node

    def process(self, bst_tree):

        try:
            lbst_tree = self.create_lbst_tree_from_bst_tree(bst_tree)
            return lbst_tree
        except RuntimeError as e:
            print('unable to parse')
            pprint(bst_tree)
            raise e

    # OperatorTree = self.create_operator_tree_from_lbst_tree(LBSTtree)
    # print(OperatorTree)
    # return

    # try:
    #     OperatorTree = self.create_operator_tree_from_lbst_tree(LBSTtree)
    #     print(OperatorTree)
    # except:
    #     print('unable to parse')
    #     print(LBSTtree)

    def create_literal_node(self, sym):
        node = {}

        if len(sym) == 2 and sym[0] == 'z':
            sym = sym[1]

        node['symbol'] = sym
        node['type'] = 'literal'

        if sym in self.OperatorList_as or sym in self.OperatorList_md or sym in self.OperatorList_eq:
            node['type'] = 'Operation'

        return node

    def create_parent_node(self, ntype):
        return {'type': ntype}

    def delete_old_child(self, tree):
        for node in tree:
            if 'child' in node:
                for child in node['child']:
                    self.delete_old_child(child)

            if 'old_child' in node:
                del node['old_child']

    def create_compound_symbol_baseline(self, bst_tree):
        compounded_tree = []
        merging = False

        numlist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ldot']

        for node in bst_tree:
            sym, clas, align = self.symbol_manager.get_symbol_from_idx(node[4])

            if (merging and sym in self.compoundable_list) or (
                    sym == 'ldot' and len(compounded_tree) > 0 and compounded_tree[-1]['symbol'][-1] in numlist):

                if compounded_tree[-1]['symbol'][-1] in numlist and sym not in numlist:
                    compounded_tree.append(self.create_literal_node(sym))
                else:
                    compounded_tree[-1]['symbol'] = compounded_tree[-1]['symbol'] + sym

                if self.count_child(node) != 0:
                    merging = False

                    compounded_tree[-1]['old_child'] = node[9]
            else:
                merging = False

                temp_node = self.create_literal_node(sym)

                temp_node['old_child'] = node[9]
                compounded_tree.append(temp_node)
                if self.count_child(node) == 0:
                    if sym in self.compoundable_list or sym in self.Prefix_compoundable_list:
                        merging = True

        return compounded_tree

    def handle_sub(self, bst_tree):
        for idx, node in enumerate(bst_tree):
            if 'old_child' not in node:
                continue

            if len(node['old_child'][6]) > 0:
                new_node = self.create_parent_node('sub')
                new_node['child'] = []
                new_node['child'].append([node])
                new_node['child'].append(self.create_lbst_tree_from_bst_tree(node['old_child'][6]))
                node['old_child'][6] = []
                new_node['old_child'] = node['old_child']

                del bst_tree[idx]
                bst_tree.insert(idx, new_node)

    def handle_sup(self, bst_tree):

        # print(BSTtree)

        for idx in range(len(bst_tree)):

            if 'old_child' not in bst_tree[idx]:
                continue

            if len(bst_tree[idx]['old_child'][5]) > 0:
                newnode = self.create_parent_node('sup')
                newnode['child'] = []
                newnode['child'].append([bst_tree[idx]])
                newnode['child'].append(self.create_lbst_tree_from_bst_tree(bst_tree[idx]['old_child'][5]))

                bst_tree[idx]['old_child'][5] = []
                newnode['old_child'] = bst_tree[idx]['old_child']

                del bst_tree[idx]
                bst_tree.insert(idx, newnode)

    def handle_sqrt(self, bst_tree):
        for idx, node in enumerate(bst_tree):
            if 'old_child' not in node:
                continue

            if len(node['old_child'][4]) > 0 and 'symbol' in node and node['symbol'] == 'sqrt':
                new_node = self.create_parent_node('sqrt')
                new_node['child'] = []
                new_node['child'].append(self.create_lbst_tree_from_bst_tree(node['old_child'][4]))
                node['old_child'][4] = []
                new_node['old_child'] = node['old_child']

                del bst_tree[idx]
                bst_tree.insert(idx, new_node)

    def handle_nonscripted_and_variablerange(self, bst_tree):
        for idx in range(len(bst_tree)):
            if 'old_child' not in bst_tree[idx] or 'symbol' not in bst_tree[idx]:
                continue

            # SUM
            # PI
            # frac
            # int
            # lim
            # rightarrow

            if len(bst_tree[idx]['old_child'][3]) > 0 or len(bst_tree[idx]['old_child'][2]) > 0:
                if bst_tree[idx]['symbol'] == '_Sigma':
                    newnode = self.create_parent_node('_Sigma')
                elif bst_tree[idx]['symbol'] == '_Pi':
                    newnode = self.create_parent_node('_Pi')
                elif bst_tree[idx]['symbol'] == 'integral ':
                    newnode = self.create_parent_node('integral')
                elif bst_tree[idx]['symbol'] == 'lim':
                    newnode = self.create_parent_node('lim')
                elif bst_tree[idx]['symbol'] == 'sub':
                    newnode = self.create_parent_node('frac')
                elif bst_tree[idx]['symbol'] == 'rightarrow':
                    newnode = self.create_parent_node('rightarrow')
                else:
                    newnode = self.create_parent_node(bst_tree[idx]['symbol'])

                newnode['child'] = []
                newnode['child'].append(self.create_lbst_tree_from_bst_tree(bst_tree[idx]['old_child'][2]))
                newnode['child'].append(self.create_lbst_tree_from_bst_tree(bst_tree[idx]['old_child'][3]))

                bst_tree[idx]['old_child'][2] = []
                bst_tree[idx]['old_child'][3] = []
                newnode['old_child'] = bst_tree[idx]['old_child']

                del bst_tree[idx]
                bst_tree.insert(idx, newnode)

    def handle_bracket(self, bst_tree):
        open_index = -1

        for idx in range(len(bst_tree)):

            if 'symbol' not in bst_tree[idx]:
                continue

            if bst_tree[idx]['symbol'] == '(':
                open_index = idx
            elif bst_tree[idx]['symbol'] == ')' and open_index > -1:
                close_index = idx

                del_list = list(range(open_index, close_index + 1))
                del_list.reverse()

                bracket_content = [bst_tree[open_index + 1: close_index]]

                newnode = self.create_parent_node('bracket')
                newnode['child'] = [self.handle_compound_tree(bracket_content[0])]

                if 'old_child' in bst_tree[idx]:
                    newnode['old_child'] = bst_tree[idx]['old_child']

                for i in del_list:
                    del bst_tree[i]

                bst_tree.insert(open_index, newnode)

                return True
        return False

    def create_lbst_tree_from_bst_tree(self, bst_tree):
        compounded_tree = self.create_compound_symbol_baseline(bst_tree)
        return self.handle_compound_tree(compounded_tree)

    def handle_compound_tree(self, compounded_tree):
        while self.handle_bracket(compounded_tree):
            pass

        self.handle_sub(compounded_tree)
        self.handle_sup(compounded_tree)

        self.handle_sqrt(compounded_tree)

        self.handle_nonscripted_and_variablerange(compounded_tree)

        self.delete_old_child(compounded_tree)

        return compounded_tree

    def create_operation_parent_node(self, ntype):
        return {'type': 'Op' + ntype}

    def create_function_parent_node(self, ntype):
        return {'type': ntype}

    def parse_to_operation_tree_binary(self, lbst_tree, op_list):
        idx_list = list(range(len(lbst_tree)))
        idx_list.reverse()

        for idx in idx_list:
            if 'symbol' not in lbst_tree[idx]:
                continue

            if lbst_tree[idx]['symbol'] in op_list:
                newnode = self.create_operation_parent_node(lbst_tree[idx]['symbol'])
                newnode['child'] = []

                newnode['child'].append([lbst_tree[idx - 1]])
                newnode['child'].append([lbst_tree[idx + 1]])

                del lbst_tree[idx + 1]
                del lbst_tree[idx]
                del lbst_tree[idx - 1]

                lbst_tree.insert(idx - 1, newnode)
                return True

        return False

    def parse_to_operation_tree_as(self, lbst_tree):
        idx_list = list(range(len(lbst_tree)))
        idx_list.reverse()

        for idx in idx_list:
            if 'symbol' not in lbst_tree[idx]:
                continue

            if lbst_tree[idx]['symbol'] in self.OperatorList_as:

                newnode = self.create_operation_parent_node(lbst_tree[idx]['symbol'])

                if idx == 0 or (
                        'symbol' in lbst_tree[idx - 1] and
                        (
                                lbst_tree[idx - 1]['symbol'] in self.OperatorList_as or
                                lbst_tree[idx - 1]['symbol'] in self.OperatorList_eq)):  # unary
                    newnode['child'] = []

                    newnode['child'].append([lbst_tree[idx + 1]])

                    del lbst_tree[idx + 1]
                    del lbst_tree[idx]

                    lbst_tree.insert(idx, newnode)
                    return True

                else:  # binary

                    newnode['child'] = []
                    newnode['child'].append([lbst_tree[idx - 1]])
                    newnode['child'].append([lbst_tree[idx + 1]])

                    del lbst_tree[idx + 1]
                    del lbst_tree[idx]
                    del lbst_tree[idx - 1]
                    lbst_tree.insert(idx - 1, newnode)
                    return True

        return False

    def add_invisible_multiply(self, lbst_tree):  # for cases such as '2a'
        idx_list = list(range(len(lbst_tree) - 1))

        for idx in idx_list:

            cond_left = lbst_tree[idx]['type'] in self.AllowAjacentAsMultiply and lbst_tree[idx][
                'symbol'] not in self.AllowAjacentAsMultiplyRight

            cond_right = lbst_tree[idx + 1]['type'] in self.AllowAjacentAsMultiply or (
                    'symbol' in lbst_tree[idx + 1] and
                    lbst_tree[idx + 1]['symbol'] in self.AllowAjacentAsMultiplyRight)

            if cond_left and cond_right:
                newnode = self.create_operation_parent_node('Otime')
                newnode['child'] = []
                newnode['child'].append([lbst_tree[idx]])
                newnode['child'].append([lbst_tree[idx + 1]])

                del lbst_tree[idx + 1]
                del lbst_tree[idx]

                lbst_tree.insert(idx, newnode)
                return True

        return False

    def parse_child(self, lbst_tree):
        idx_list = list(range(len(lbst_tree)))

        for idx in idx_list:

            if 'child' not in lbst_tree[idx]:
                continue

            for i in range(len(lbst_tree[idx]['child'])):
                lbst_tree[idx]['child'][i] = self.parse_to_operation_tree(lbst_tree[idx]['child'][i])

    def parse_function_operator(self, lbst_tree):  # for cases such as f(x)
        idx_list = list(range(len(lbst_tree) - 1))
        idx_list.reverse()

        for idx in idx_list:

            cond_function_name = 'symbol' in lbst_tree[idx] and lbst_tree[idx][
                'symbol'] in self.AllowAjacentAsMultiplyRight
            cond_user_defined_name = 'symbol' in lbst_tree[idx] and lbst_tree[idx][
                'symbol'] not in self.AllOp  # not merge to cond_function_name because i am not so sure about this case

            cond_left = cond_user_defined_name or cond_function_name or lbst_tree[idx]['type'] in self.VariableRangeList

            type_right = lbst_tree[idx + 1]['type']
            if type_right in self.AllOp or (len(type_right) > 1 and type_right[:2] == 'Op'):
                continue

            if cond_left:
                if cond_function_name or cond_user_defined_name:
                    newnode = self.create_function_parent_node('f' + lbst_tree[idx]['symbol'])
                else:
                    newnode = self.create_operation_parent_node('f' + lbst_tree[idx]['type'])
                newnode['child'] = []

                if 'child' in lbst_tree[idx]:
                    for c in lbst_tree[idx]['child']:
                        newnode['child'].append(c)

                newnode['child'].append([lbst_tree[idx + 1]])

                del lbst_tree[idx + 1]
                del lbst_tree[idx]

                lbst_tree.insert(idx, newnode)
                return True

        return False

    def parse_to_operation_tree(self, lbst_tree):

        self.parse_child(lbst_tree)

        while self.add_invisible_multiply(lbst_tree):
            pass

        while self.parse_function_operator(lbst_tree):
            pass

        while self.parse_to_operation_tree_binary(lbst_tree, self.OperatorList_md):
            pass
        while self.parse_to_operation_tree_as(lbst_tree):
            pass
        while self.parse_to_operation_tree_binary(lbst_tree, self.OperatorList_eq):
            pass

        return lbst_tree

    def create_operator_tree_from_lbst_tree(self, lbst_tree):

        return self.parse_to_operation_tree(lbst_tree)

    def create_lbst_node_from_bst_node(self, bst_node):
        # sym, clas, align = self.symbol_manager.get_symbol_from_idx(BSTnode[4])
        self.symbol_manager.get_symbol_from_idx(bst_node[4])

    def count_child(self, bst_node):
        childnodes = bst_node[9]
        count = 0
        for child in childnodes:
            count += len(child)

        return count


class LatexGenerator:
    def __init__(self):
        self.greek_alphabet = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'muy', 'rho', 'nuy', 'omega', 'lamda', 'phi', 'sigma', 'theta', 'pi']

    def process(self, lbst_tree):
        output = self.create_latex_string(lbst_tree)
        # print(output)
        return output

    def get_string_from_node(self, node):
        if 'type' in node:
            if node['type'] == 'literal':
                if node['symbol'] == 'integral':
                    return '\\int'
                if node['symbol'] == '_Delta':
                    return '\\Delta'
                if node['symbol'] == 'ldots':
                    return '\\ldots'
                if node['symbol'] == 'rightarrow':
                    return ' \\rightarrow '
                if node['symbol'] == 'intft':
                    return ' \\infty '

                node['symbol'] = node['symbol'].replace('ldot', '.')

                if node['symbol'] in self.greek_alphabet:
                    if node['symbol'] == 'muy':
                        node['symbol'] = '\\mu'
                    else:
                        node['symbol'] = '\\' + node['symbol']

                return ' ' + node['symbol'] + ' '
            elif node['type'] == 'Operation':
                if node['symbol'] == 'add':
                    return ' + '
                if node['symbol'] == 'time':
                    return '\\times '
                if node['symbol'] == 'sub':
                    return ' - '
                if node['symbol'] == 'neq':
                    return ' \\neq '
                if node['symbol'] == 'geq':
                    return ' \\geq '
                if node['symbol'] == 'leq':
                    return ' \\leq '
                if node['symbol'] == 'in':
                    return ' \\in '
                if node['symbol'] == 'div':
                    return ' \\div '
                if node['symbol'] == 'rightarrow':
                    return ' \\rightarrow '

                return node['symbol']
            elif node['type'].startswith('bracket'):
                tmp_bracket = ['(']
                if '(' in tmp_bracket:
                    tmp_bracket.append(')')
                elif '[' in tmp_bracket:
                    tmp_bracket.append(']')
                elif '\\{' in tmp_bracket:
                    tmp_bracket.append('\\}')
                return tmp_bracket[0] + self.create_latex_string(node['child'][0]) + tmp_bracket[1]
            elif node['type'] == 'sqrt':
                return '\\sqrt{' + self.create_latex_string(node['child'][0]) + '}'
            elif node['type'] == 'sup':

                base = self.create_latex_string(node['child'][0])
                sup = self.create_latex_string(node['child'][1])

                return base + '^{ ' + sup + ' }'

            elif node['type'] == 'sub':

                base = self.create_latex_string(node['child'][0])
                sub = self.create_latex_string(node['child'][1])

                # SPECIAL #############
                if 'symbol' in node['child'][1][0] and node['child'][1][0]['symbol'] == '.':
                    return base + sub
                ##################################

                return base + '_{ ' + sub + ' }'

            elif node['type'] == 'frac':
                upper_node = self.create_latex_string(node['child'][0])
                lower_node = self.create_latex_string(node['child'][1])

                return '\\frac{' + upper_node + '}{' + lower_node + '}'

            elif node['type'] == '_Sigma':
                upper_node = self.create_latex_string(node['child'][0])
                lower_node = self.create_latex_string(node['child'][1])

                return '\\sum^{' + upper_node + '}_{' + lower_node + '}'

            elif node['type'] == '_Pi':
                upper_node = self.create_latex_string(node['child'][0])
                lower_node = self.create_latex_string(node['child'][1])

                return '\\prod^{' + upper_node + '}_{' + lower_node + '}'

            elif node['type'] == 'lim':
                lower_node = self.create_latex_string(node['child'][1])

                return '\\lim_{' + lower_node + '}'

            elif node['type'] == 'integral':
                upper_node = self.create_latex_string(node['child'][0])
                lower_node = self.create_latex_string(node['child'][1])

                return '\\int^{' + upper_node + '}_{' + lower_node + '}'
            else:
                if len(node['child'][0]) > 0 and len(node['child'][1]) == 0:
                    return node['type'] + '^{' + self.create_latex_string(node['child'][0]) + '}'

                elif len(node['child'][0]) == 0 and len(node['child'][1]) > 0:
                    return node['type'] + '_{' + self.create_latex_string(node['child'][1]) + '}'

                elif len(node['child'][0]) > 0 and len(node['child'][1]) > 0:
                    return node['type'] + '^{' + self.create_latex_string(
                        node['child'][0]) + '}_{' + self.create_latex_string(node['child'][1]) + '}'

                return str(node)
        else:
            return str(node)

    def create_latex_string(self, lbst_tree):
        latex_str = ''

        if type(lbst_tree) == dict:

            latex_str = latex_str + self.get_string_from_node(lbst_tree)

            if 'child' in lbst_tree:
                pass
            return latex_str

        for child in lbst_tree:
            latex_str = latex_str + self.create_latex_string(child)

        return latex_str


class BBParser:
    def __init__(self):
        self.Atb = {
            'tlx': 0,
            'tly': 1,
            'brx': 2,
            'bry': 3,
            'label': 4,
            'centroidx': 5,
            'centroidy': 6,
            'thres_sub': 7,
            'thres_sup': 8,
            'child_temp': 9,
            'child_main': 10
        }
        self.ChildLabel = {
            'TL': 0,
            'BL': 1,
            'T': 2,
            'B': 3,
            'C': 4,
            'SUP': 5,
            'SUB': 6
        }

        self.debugInt = 0
        # print('initializing ...')

        self.symbol_manager = SymbolManager()

        self.latexgenerator = LatexGenerator()
        # self.threshold_ratio_t = 0.9
        self.threshold_ratio_t = 1 - 1/6
        # self.centroid_ratio_c = 0.5
        self.centroid_ratio_c = 1 - 1/4

        # debug:
        self.handling_file = ''

    # region_label
    # TL TR AB BL
    #

    def get_data_from_file(self):  # debug
        # with open('Exp_train_giaidoan2.txt') as f:
        with open('ssd_train.txt') as f:
            z = f.readlines()
            self.test_candidate = z

    def debug(self):
        # self.process('training/data/CROHME_2013_valid/rit_42160_4.png 92 48.06818181818182 123.9255485893417 75.52115987460816 177.13087774294672 9 71.14811912225706 147.7343260188088 92.2844827586207 148.70611285266457 69 75.27821316614421 154.5368338557994 87.18260188087774 181.98981191222572 18 82.5666144200627 121.01018808777428 83.0525078369906 137.04467084639498 8 99.0869905956113 143.60423197492162 110.262539184953 158.1810344827586 9 116.57915360501568 148.22021943573668 152.29231974921632 150.16379310344828 18 118.76567398119124 155.26567398119124 119.98040752351099 172.51489028213166 9 126.29702194357367 163.2829153605016 135.52899686520377 163.52586206896552 18 136.3793103448276 127.32680250783699 138.07993730407526 144.33307210031347 69 138.4443573667712 155.99451410658307 149.86285266457682 178.83150470219437 48 163.71081504702195 133.6434169278997 173.185736677116 158.1810344827586 69 173.185736677116 143.84717868338558 186.30485893416932 170.08542319749216 13 193.1073667711599 146.27664576802508 202.8252351097179 152.35031347962382 92 207.44122257053294 127.32680250783699 228.33463949843264 169.11363636363637 48 228.33463949843264 127.56974921630095 246.06974921630098 155.99451410658307 68 247.28448275862073 143.36128526645768 259.4318181818182 157.20924764890282')
        # self.process('null 4 272 236 323 287 13 143 265 191 323 5 359 202 438 280 72 108 173 169 258 72')
        # self.process(
        #     'null '
        #     '60 48.06818181818182 137.0121849661801 63.2345830362556 180.47563006763986 '
        #     '1 83.77147385123423 122.52436993236017 93.4050448396467 158.74390751690999 '
        #     '89 100.64895235655666 129.76827744927013 137.81100578903155 158.74390751690999 '
        #     '1 159.30111799179582 122.52436993236017 167.02824620463701 158.74390751690999 '
        #     '23 182.25672081317646 122.52436993236017 202.50712421446732 158.74390751690999 '
        #     '2 224.54067624506845 122.52436993236017 231.18092480223592 158.74390751690999 '
        #     '2 253.6373840410234 122.52436993236017 259.4318181818182 158.74390751690999')  # brackets
        # self.process(
        #     'tmp.txt.png '
        #     '50 48.06818181818181 117.48819327601055 57.449244109329044 150.13952773104043 '
        #     '13 62.89113318516736 127.64061325242594 79.2168004126823 142.7080522925442 '
        #     '9 84.65868948852062 120.57965106606514 259.43181818181813 149.769014478905 '
        #     '22 135.47135911035906 105.24394285537434 159.13174639661258 132.45338823456592 '
        #     '22 169.96481992398586 105.24394285537434 186.39189640010758 132.45338823456592 '
        #     '55 198.7088096781656 105.24394285537434 212.16820665291772 132.45338823456592 '
        #     '26 105.38796878632992 154.22094453791917 120.26771035211914 181.43038991711074 '
        #     '72 125.70959942795746 165.1047226895958 141.97149451786493 181.43038991711074 '
        #     '61 147.4133835937032 165.1047226895958 165.4081433924821 197.7560571446257 '
        #     '11 170.85003246832042 159.6628336137575 186.55568859770995 175.9885008412724 '
        #     '25 195.73899020310589 154.22094453791917 215.46561052318222 181.43038991711074 '
        #     '61 220.90749959902053 165.1047226895958 251.0322426974112 197.7560571446257')  # frac
        # self.process('training/data/CROHME_2013_valid/122_em_358.png 2 0.9997619986534119 234.9671173095703 93.77392578125 258.2026062011719 201.8711395263672 1 0.9866085648536682 152.92860412597656 103.67342376708984 171.04006958007812 204.31536865234375 48 0.9825618267059326 79.68318939208984 152.41653442382812 121.00485229492188 223.09934997558594 65 0.9349151849746704 179.47471618652344 112.53890228271484 229.4780731201172 173.55006408691406 60 0.37060970067977905 44.50164794921875 75.71089172363281 104.27759552001953 202.2056884765625', True)
        # self.process('training/data/CROHME_2013_valid/122_em_358.png 4 0.9997619986534119 234.9671173095703 93.77392578125 258.2026062011719 201.8711395263672 3 0.9866085648536682 152.92860412597656 103.67342376708984 171.04006958007812 204.31536865234375 48 0.9825618267059326 79.68318939208984 152.41653442382812 121.00485229492188 223.09934997558594 65 0.9349151849746704 179.47471618652344 112.53890228271484 229.4780731201172 173.55006408691406 60 0.37060970067977905 44.50164794921875 75.71089172363281 104.27759552001953 202.2056884765625', True)
        # self.process('training/data/CROHME_2013_valid/122_em_358.png 6 0.9997619986534119 234.9671173095703 93.77392578125 258.2026062011719 201.8711395263672 5 0.9866085648536682 152.92860412597656 103.67342376708984 171.04006958007812 204.31536865234375 48 0.9825618267059326 79.68318939208984 152.41653442382812 121.00485229492188 223.09934997558594 65 0.9349151849746704 179.47471618652344 112.53890228271484 229.4780731201172 173.55006408691406 60 0.37060970067977905 44.50164794921875 75.71089172363281 104.27759552001953 202.2056884765625', True)
        # self.process('training/data/CROHME_2013_valid/122_em_358.png 2 0.9997619986534119 234.9671173095703 93.77392578125 258.2026062011719 201.8711395263672 1 0.9866085648536682 152.92860412597656 103.67342376708984 171.04006958007812 204.31536865234375 26 0.9825618267059326 79.68318939208984 152.41653442382812 121.00485229492188 223.09934997558594 74 0.9349151849746704 179.47471618652344 112.53890228271484 229.4780731201172 173.55006408691406 57 0.37060970067977905 44.50164794921875 75.71089172363281 104.27759552001953 202.2056884765625', True)

        # DEBUG
        # self.process(
        #     'training/data/CROHME_2013_test/103_em_17.png 2 0.9992454051971436 240.3772430419922 121.35649108886719 258.79339599609375 178.01254272460938 13 0.9877486824989319 68.69273376464844 163.77598571777344 91.6712875366211 177.43063354492188 41 0.9642698764801025 186.72311401367188 135.37887573242188 219.9486541748047 167.43820190429688 30 0.8630313873291016 126.20185089111328 136.44190979003906 150.197265625 180.09600830078125 59 0.5575922727584839 221.53233337402344 162.30902099609375 232.98114013671875 178.04737854003906 18 0.3431088924407959 50.41469955444336 156.61282348632812 52.51301574707031 175.2626190185547',
        #     is_pred=True)
        # self.process(
        #     'training/data/CROHME_2013_test/105_em_84.png 33 0.9933397173881531 132.62306213378906 131.38827514648438 161.86477661132812 171.61778259277344 36 0.8074544072151184 236.60617065429688 128.48800659179688 259.1343994140625 165.55169677734375 33 0.4563315510749817 51.17082595825195 129.5625762939453 79.73133850097656 185.4637451171875 18 0.25306305289268494 178.65350341796875 110.41378784179688 185.0289764404297 127.48966979980469 16 0.16595715284347534 204.68557739257812 170.8423309326172 206.23019409179688 177.05999755859375 19 0.12819036841392517 104.23164367675781 177.05508422851562 111.16339874267578 192.19247436523438',
        #     is_pred=True)
        self.process(
            'training/data/CROHME_2013_test/116_em_171.png 2 0.9462464451789856 89.13607025146484 141.98289489746094 95.29906463623047 163.86849975585938 34 0.8820258975028992 132.25820922851562 139.7476348876953 151.8017578125 160.53799438476562 1 0.8614202737808228 74.5179443359375 140.3698272705078 79.48596954345703 166.79734802246094 100 0.7768757939338684 166.94268798828125 131.65354919433594 194.4500274658203 159.97549438476562 73 0.7484859824180603 206.37820434570312 152.520263671875 228.64389038085938 163.0619354248047 63 0.4680389165878296 253.61431884765625 159.5491180419922 260.52545166015625 169.7353973388672 20 0.45515379309654236 151.6776123046875 159.3052520751953 157.28372192382812 169.0990753173828 89 0.37159931659698486 233.8389129638672 149.49293518066406 249.115478515625 162.04356384277344 13 0.34236404299736023 109.96321868896484 152.04701232910156 119.18383026123047 158.589111328125 19 0.2276642769575119 172.34934997558594 136.45407104492188 190.04083251953125 158.04647827148438 58 0.11929110437631607 47.878265380859375 147.58290100097656 57.358917236328125 156.97499084472656',
            is_pred=True)

    def process(self, input_line, is_pred=False):  # input is raw string
        print(input_line)
        raw_line = input_line.strip().split()

        self.handling_file = raw_line[0]
        raw_line = raw_line[1:]

        raw_line = list(map(lambda s: int(float(s)), raw_line))

        bbox_lst = []

        if not is_pred:
            for i in range(int(len(raw_line) / 5)):
                bbox_lst.append(raw_line[5 * i + 1: 5 * (i + 1)] + [raw_line[5 * i]])
        else:
            for i in range(int(len(raw_line)/6)):
                bbox_lst.append(raw_line[6*i + 2:6*(i+1)] + [raw_line[6*i]])

        self.preprocess_bbox_lst(bbox_lst)
        bst = self.build_bst(bbox_lst)
        # pprint(bst)
        # self.debug_print(bst)

        self.current_LBST = LBST(self.symbol_manager)

        lbst_tree = self.current_LBST.process(bst)

        latex_string = self.latexgenerator.process(lbst_tree)

        print(latex_string)
        return latex_string

    # BASE PROCESS
    # def process(self, input):  # input is raw string
    # 	raw_line = input.replace('\n', '').split(' ')
    #
    # 	self.handling_file = raw_line[0]
    # 	raw_line = raw_line[1:]
    #
    # 	raw_line = list(map(lambda s: int(float(s)), raw_line))
    #
    # 	BB_List = []
    #
    # 	for i in range(raw_line[0]):
    # 		BB_List.append(raw_line[5 * i + 1: 5 * i + 6])
    #
    # 	self.preprocess_bbox_lst(BB_List)
    # 	BST = self.build_bst(BB_List)
    # 	# self.debug_print(BST)
    #
    # 	self.current_LBST = LBST(self.symbol_manager)
    #
    # 	LBSTTree = self.current_LBST.process(BST)
    #
    # 	latex_string = self.latexgenerator.process(LBSTTree)
    #
    # 	return latex_string

    def build_bst(self, bbox_lst):
        """
        Build BST tree from a list of bboxes
        :param bbox_lst: each bbox has the following format : (tlx, tly, brx, bry, label, centroidx, centroidy, thres_sub, thres_sup, child)
        :return:
        """
        bst = []
        if len(bbox_lst) == 0:
            return bst
        node_list = sorted(bbox_lst, key=itemgetter(0))  # sort bboxes by xmin

        retvalue = self.extract_baseline(node_list)

        return retvalue

    def debug_print(self, tree):
        # print(self.handling_file)
        for i in tree:
            sym, clas, align = self.symbol_manager.get_symbol_from_idx(i[4])
            # print(i)
            print(sym)
            for j in i[9]:
                print(j)
            print('---')

    def extract_baseline(self, rnode_list):
        if len(rnode_list) < 1:
            return rnode_list

        # get baseline starting symbol
        s_start = self.starting_baseline_symbol(rnode_list)

        # infer s_start index
        idx = 0
        for i in rnode_list:
            if i[0] == s_start[0] and i[1] == s_start[1] and i[4] == s_start[4]:
                break
            idx += 1

        # remove s_start from rnode
        del rnode_list[idx]

        baseline_symbols = self.hor([s_start], rnode_list)

        updated_baseline = self.collect_region(baseline_symbols)

        for symbol in updated_baseline:
            for idx in range(len(symbol[self.Atb['child_temp']])):
                temp = self.extract_baseline(symbol[self.Atb['child_temp']][idx])

                # print('>>>>>')
                # print(symbol[self.Atb['child_temp']][idx])
                # print('---------')
                # print(temp)
                # print('<<<<<')
                symbol[self.Atb['child_temp']][idx] = temp
                pass

        return updated_baseline

    def starting_baseline_symbol(self, snode_lst):
        """
        find first symbol in baseline

        :param snode_lst:
        :return:
        """

        if len(snode_lst) < 1:
            return 0

        temp_list = snode_lst[:]
        while len(temp_list) > 1:

            sym_n = temp_list[-1]
            sym_n_1 = temp_list[-2]

            sym_n_sym, sym_n_class, sym_n_align = self.symbol_manager.get_symbol_from_idx(sym_n[4])

            if self.overlap(sym_n, sym_n_1) \
                    or self.contains(sym_n, sym_n_1) \
                    or (sym_n_class == 'VariableRange' and not self.is_adjacent(sym_n_1, sym_n)):
                # s_n dominate
                del temp_list[-2]
            else:
                del temp_list[-1]

        return temp_list[0]

    # self.debug_draw(temp_list)

    def hor(self, snode_lst_1, snode_lst_2):
        """
        Find the symbols of the baseline that begins with the symbols in snode_list_1
        and continues with a subset of the symbols in snode_list_2.
        The symbols of the baseline are returned as snode_list'.
        Nonbaseline symbols in snode_list_2 are partitioned into TLEFT, BLEFT, ABOVE, BELOW, and CONTAINS regions.
        Symbols in TLEFT and BLEFT regions are later reassigned by the collect_regions function.
        :param snode_lst_1: list contains only starting node
        :param snode_lst_2: list of other nodes
        :return:
        """
        if len(snode_lst_2) == 0:
            return snode_lst_1

        current_symbol = snode_lst_1[-1]

        remaining_symbols, current_symbol_new = self.partition(snode_lst_2, copy.deepcopy(current_symbol))

        # replace

        snode_lst_1.pop()

        snode_lst_1.append(current_symbol_new)

        if len(remaining_symbols) == 0:
            return snode_lst_1

        # 6
        sym, clas, align = self.symbol_manager.get_symbol_from_idx(current_symbol_new[4])

        if clas == 'NonScripted':
            temp = self.starting_baseline_symbol(remaining_symbols)
            temp = snode_lst_1 + [temp]

            return self.hor(temp, remaining_symbols)

        sl = remaining_symbols[:]

        while len(sl) > 0:
            l1 = sl[0]

            if self.is_regular_hor(current_symbol_new, l1):
                ####
                # if (len(sl) == 2):

                temp = self.check_overlap(l1, remaining_symbols)

                temp = snode_lst_1 + [temp]

                return self.hor(temp, remaining_symbols)

            sl = sl[1:]

        current_symbol_new = self.partition_final(remaining_symbols, copy.deepcopy(current_symbol_new))

        temp = snode_lst_1[:]
        temp.pop()
        temp.append(current_symbol_new)

        return temp

    def collect_region(self, snode_list):
        temp = self.collect_region_partial(snode_list, 'TL')
        return self.collect_region_partial(temp, 'BL')

    def collect_region_partial(self, snode_list, region):
        if len(snode_list) == 0:
            return snode_list

        if region == 'TL':
            region_diagonal = 'TL'
            region_s1_diagonal = 'SUP'
            region_list = ['TL', 'T', 'SUP']
            region_vertical = 'T'
        else:
            region_diagonal = 'BL'
            region_s1_diagonal = 'SUB'
            region_list = ['BL', 'B', 'SUB']
            region_vertical = 'B'

        s1 = copy.deepcopy(snode_list[0])
        s1_new = copy.deepcopy(s1)

        snode_list_new = snode_list[1:]

        if len(snode_list) > 1:
            s2 = snode_list[1]
            super_list, tleft_list = self.partition_shared_region(region_diagonal, s1, s2)
            s1_new = self.add_region(region_s1_diagonal, super_list, s1)
            s2_new = self.add_region(region_diagonal, tleft_list, self.remove_region([region_diagonal], s2))

            del_idx = 0
            for i in snode_list_new:
                if i[0] == s2[0] and i[1] == s2[1] and i[4] == s2[4]:
                    break
                del_idx = del_idx + 1
            del snode_list[del_idx]

            snode_list.insert(del_idx, s2_new)

        syms1_new, class1_new, aligs1_new = self.symbol_manager.get_symbol_from_idx(s1_new[self.Atb['label']])
        if class1_new == 'VariableRange':
            # region_list = ['TL', 'T', 'SUP']

            s1_new = self.merge_region(region_list, region_vertical, s1)

        return [s1_new] + self.collect_region(snode_list_new)

    def is_regular_hor(self, snode1, snode2):
        # sym1, clas1, alig1 = self.symbol_manager.get_symbol_from_idx(snode1[self.Atb['label']])
        self.symbol_manager.get_symbol_from_idx(snode1[self.Atb['label']])
        sym2, clas2, alig2 = self.symbol_manager.get_symbol_from_idx(snode2[self.Atb['label']])

        cond_a = self.is_adjacent(snode2, snode1)

        cond_b = snode1[1] > snode2[1] and snode1[3] < snode2[3]  # 1 in 2 horizontally
        cond_c = (sym2 == '(' or sym2 == ')') and snode2[1] < snode1[6] < snode2[3]

        return cond_a or cond_b or cond_c

    def preprocess_bbox_lst(self, bbox_lst):
        # format: (tlx, tly, brx, bry, label, centroidx, centroidy, thres_sub, thres_sup, child)
        # child_temp: TL, BL, T, B, C, SUP, SUB
        # child_main: SUP, SUB, UPP, LOW
        for bbox in bbox_lst:
            sym, clas, align = self.symbol_manager.get_symbol_from_idx(bbox[4])

            height = bbox[3] - bbox[1]

            # centroid x  # 5
            bbox.append((bbox[0] + bbox[2]) / 2)

            # centroid y  # 6
            if align == 'Centred':
                # c = bbox[1] + (bbox[3] - bbox[1]) / 2.0
                cy = bbox[1] + height / 2
            elif align == 'Ascender':
                # c = bbox[1] + (bbox[3] - bbox[1]) / 4.0 * 3
                cy = bbox[1] + height * self.centroid_ratio_c
            else:
                # c = bbox[1] + (bbox[3] - bbox[1]) / 4.0
                cy = bbox[1] + height * (1 - self.centroid_ratio_c)
            bbox.append(cy)

            # thres_sub  # 7
            # thres_sup  # 8
            # if clas == 'NonScripted':
            #     bbox.append(bbox[6])
            #     bbox.append(bbox[6])
            # elif clas == 'Bracket':
            # elif clas == 'Plain_Descender':
            if clas == 'Plain_Descender':
                # bbox.append(bbox[1] + 0.5 * height + 0.5 * height * self.threshold_ratio_t)
                # bbox.append(bbox[1] + height - 0.5 * height * self.threshold_ratio_t)
                t_sub = bbox[1] + height / 2 + height * self.threshold_ratio_t / 2
                t_sup = bbox[1] + height - height * self.threshold_ratio_t / 2
            else:
                # bbox.append(bbox[1] + height * self.threshold_ratio_t)
                # bbox.append(bbox[1] + height - height * self.threshold_ratio_t)
                t_sub = bbox[1] + height * self.threshold_ratio_t
                t_sup = bbox[1] + height - height * self.threshold_ratio_t
            bbox.append(t_sub)
            bbox.append(t_sup)

            # child tmp
            bbox.append([[], [], [], [], [], [], []])

            # sym1, clas1, alig1 = self.symbol_manager.get_symbol_from_idx(bbox[4])
            # bbox.append(sym1)
            bbox.append(sym)

            # thres below  # 11
            # thres above  # 12
            if clas == 'NonScripted':
                thres_b = thres_a = bbox[1] + height / 2
            elif clas in ['Bracket', 'Root']:
                thres_b = bbox[3]
                thres_a = bbox[1]
            elif clas == 'Plain_Descender':
                thres_b = bbox[1] + height / 2 + height * self.threshold_ratio_t / 2
                thres_a = bbox[1] + height - height * self.threshold_ratio_t / 2
            else:
                thres_b = bbox[1] + height * self.threshold_ratio_t
                thres_a = bbox[1] + height - height * self.threshold_ratio_t
            bbox.append(thres_b)
            bbox.append(thres_a)

            # print(bbox)

    # BB.append([[], [], [], []])

    def overlap(self, snode_1, snode_2):
        """
        Test whether snode_1 is a Nonscripted symbol that vertically overlaps snode_2.

        tested on 5 criterias:
        + snode_1 class is NonScripted    (cond_b)
        + snode_2 centroid_x in between snode_1 x_min x_max    (cond_c)
        + snode_2 does NOT contain snode_1    (cond_d)
        + NOT snode_2 is either ( or ) and snode_2 contains snode_1    (cond_e_i)
        + NOT snode_2 is either NonScripted or VariableRange and snode_2 is wider than snode_1

        :param snode_1: n
        :param snode_2: n - 1
        :return:
        """

        if snode_1[0] == snode_2[0] and snode_1[1] == snode_2[1]:
            return False

        sym1, clas1, alig1 = self.symbol_manager.get_symbol_from_idx(snode_1[4])
        sym2, clas2, alig2 = self.symbol_manager.get_symbol_from_idx(snode_2[4])

        cond_b = clas1 == 'NonScripted'
        cond_c = snode_1[0] <= snode_2[5] < snode_1[2]
        cond_d = not self.contains(snode_2, snode_1)

        cond_e_i = \
            (sym2 == '(' or sym2 == ')') and \
            (snode_2[1] <= snode_1[6] < snode_2[1]) and \
            (snode_2[0] <= snode_1[5] < snode_2[2])

        cond_e_ii = (clas2 == 'NonScripted' or clas2 == 'VariableRange') and (
                snode_2[2] - snode_2[0] > snode_1[2] - snode_1[0])

        cond_e = (not cond_e_i) and (not cond_e_ii)

        ret = cond_b and cond_c and cond_d and cond_e
        return ret

    def check_overlap(self, snode, snode_list):
        longest = -1

        return_candidate = 0

        for node in snode_list:
            sym, clas, alig = self.symbol_manager.get_symbol_from_idx(node[4])
            if clas == 'NonScripted' and self.overlap(node, snode):
                w = node[2] - node[0]
                if w > longest:
                    return_candidate = node
                    longest = w

        if longest == -1:
            return snode
        else:
            return return_candidate

    def contains(self, snode_1, snode_2):
        """
        check whether snode_1 is a sqrt and contains snode_2

        3 criterias:
        + snode_1 is a sqrt
        + snode_2 centroid_x in between snode_1 x_min x_max
        + snode_2 centroid_y in between snode_1 y_min y_max

        :param snode_1:
        :param snode_2:
        :return:
        """
        sym1, clas1, alig1 = self.symbol_manager.get_symbol_from_idx(snode_1[4])
        # sym2, clas2, alig2 = self.symbol_manager.get_symbol_from_idx(snode_2[4])

        cond_1 = clas1 == 'Root'  # if snode_1 is a sqrt
        cond_2 = snode_1[0] <= snode_2[5] < snode_1[2]
        cond_3 = snode_1[1] <= snode_2[6] < snode_1[4]
        return cond_1 and cond_2 and cond_3

    def is_adjacent(self, snode_1, snode_2):  # Test whether snode1 is horizontally adjacent to snode2, where snode1 may be to the left or right of snode2
        # sym1, clas1, alig1 = self.symbol_manager.get_symbol_from_idx(snode_1[4])
        self.symbol_manager.get_symbol_from_idx(snode_1[4])
        sym2, clas2, alig2 = self.symbol_manager.get_symbol_from_idx(snode_2[4])

        # format: (tlx, tly, brx, bry, label, centroidx, centroidy, thres_sub, thres_sup)
        # child: TL, BL, T, B, C
        # print('aa')
        # print(snode_1)
        # print(snode_2)
        # print('-----')
        # print(str(snode_2[7]) + ' > ' + str(snode_1[6]) )
        # print(str(snode_1[6]) + ' > ' + str(snode_2[8]) )

        return (not is_the_same(snode_1, snode_2)) and (clas2 != 'NonScripted') and (snode_2[7] > snode_1[6] > snode_2[8])

    def partition(self, snode_list, snode):
        temp_list = snode_list[:]

        del_idx_list = []

        for idx, node in enumerate(temp_list):
            if is_the_same(node, snode):  # same symbol
                del_idx_list.append(idx)
            elif node[5] < snode[0]:  # node_centroidX < snode_minX  # partitions into TLEFT BLEFT regions using SUPER SUBSC thresholds
                if node[6] < snode[8]:  # node_centroidY < snode_sup # Topleft
                    snode[self.Atb['child_temp']][0].append(node[:])
                    del_idx_list.append(idx)
                elif node[6] > snode[7]:  # node_centroidY > snode_sub # Botleft
                    snode[self.Atb['child_temp']][1].append(node[:])
                    del_idx_list.append(idx)
            elif node[5] < snode[2]:  # node_centroidX < snode_maxX  # partitions into ABOVE BELOW regions using ABOVE BELOW thresholds
                if node[6] < snode[12]:  # ABOVE
                    snode[self.Atb['child_temp']][2].append(node[:])
                    del_idx_list.append(idx)
                elif node[6] > snode[11]:  # BELOW
                    snode[self.Atb['child_temp']][3].append(node[:])
                    del_idx_list.append(idx)
                else:  # Contain
                    # if node[0] != snode[0] or node[1] != snode[1] or node[4] != snode[4]:
                    snode[self.Atb['child_temp']][4].append(node[:])
                    del_idx_list.append(idx)
            # else:
            #     if node[0] == snode[0] and node[1] == snode[1] and node[4] == snode[4]:
            #         del_idx_list.append(idx)

        for idx in del_idx_list[::-1]:
            del temp_list[idx]

        return temp_list, snode

    def partition_final(self, snode_list, snode):
        # format: (tlx, tly, brx, bry, label, centroidx, centroidy, thres_sub, thres_sup)
        # child: TL, BL, T, B, C, SUP, SUB
        for node in snode_list:
            if node[6] < snode[8]:  # centroid < sup
                snode[self.Atb['child_temp']][5].append(node[:])
            else:
                snode[self.Atb['child_temp']][6].append(node[:])

        return snode

    def partition_shared_region(self, region_label, snode1, snode2):

        s_node_lst_1 = []
        s_node_lst_2 = []

        idx = 0
        if region_label == 'BL':
            idx = 1

        # rnode = sl = copy.deepcopy(snode2[self.Atb['child_temp']][idx])
        sl = copy.deepcopy(snode2[self.Atb['child_temp']][idx])

        sym1, clas1, alig1 = self.symbol_manager.get_symbol_from_idx(snode1[4])
        sym2, clas2, alig2 = self.symbol_manager.get_symbol_from_idx(snode2[4])

        if clas1 == 'Nonscripted':
            s_node_lst_1 = []
            return s_node_lst_1, sl

        elif (clas2 != 'VariableRange') or (clas2 == 'VariableRange' and not self.has_non_empty_region(snode2, 'T')):
            s_node_lst_1 = sl
            return s_node_lst_1, s_node_lst_2

        elif clas2 == 'VariableRange' and self.has_non_empty_region(snode2, 'T'):
            for i in sl:
                if self.is_adjacent(i, snode2):
                    s_node_lst_1.append(i)
                else:
                    s_node_lst_2.append(i)

            return s_node_lst_1, s_node_lst_2

    def has_non_empty_region(self, snode, region_label):
        # child: TL, BL, T, B, C, SUP, SUB
        return len(snode[self.Atb['child_temp']][self.ChildLabel[region_label]]) > 0

    # if region_label == 'TL':
    # 	return len(snode[self.Atb['child_temp']][0]) > 0
    # elif region_label == 'BL'
    # 	return len(snode[self.Atb['child_temp']][1]) > 0
    # elif region_label == 'T'
    # 	return len(snode[self.Atb['child_temp']][2]) > 0
    # elif region_label == 'B'
    # 	return len(snode[self.Atb['child_temp']][3]) > 0
    # elif region_label == 'C'
    # 	return len(snode[self.Atb['child_temp']][4]) > 0
    # elif region_label == 'SUP'
    # 	return len(snode[self.Atb['child_temp']][5]) > 0
    # elif region_label == 'SUB'
    # 	return len(snode[self.Atb['child_temp']][6]) > 0
    # return False

    def debug_draw(self, temp_list):
        # print('debug draw')
        # sym1, clas1, alig1 = self.symbol_manager.get_symbol_from_idx(temp_list[0][4])

        self.symbol_manager.get_symbol_from_idx(temp_list[0][4])

        img = Image.open('./hardimg/' + self.handling_file)

        # fnt = ImageFont.truetype('./font/arial.ttf', 40)
        draw = ImageDraw.Draw(img)

        draw.rectangle(list(temp_list[0][:4]), outline='red')

        img.save('./result2/' + self.handling_file)

    def add_region(self, region_label, list_to_add, snode):
        snode[self.Atb['child_temp']][self.ChildLabel[region_label]] = \
            snode[self.Atb['child_temp']][self.ChildLabel[region_label]] + list_to_add

        # if region_label == 'TL':
        # 	snode[self.Atb['child_temp']][0] = snode[self.Atb['child_temp']][0] + list_to_add
        # elif region_label == 'BL'
        # 	snode[self.Atb['child_temp']][1] = snode[self.Atb['child_temp']][1] + list_to_add
        # elif region_label == 'T'
        # 	snode[self.Atb['child_temp']][2] = snode[self.Atb['child_temp']][2] + list_to_add
        # elif region_label == 'B'
        # 	snode[self.Atb['child_temp']][3] = snode[self.Atb['child_temp']][3] + list_to_add
        # elif region_label == 'C'
        # 	snode[self.Atb['child_temp']][4] = snode[self.Atb['child_temp']][4] + list_to_add
        # elif region_label == 'SUP'
        # 	snode[self.Atb['child_temp']][5] = snode[self.Atb['child_temp']][5] + list_to_add
        # elif region_label == 'SUB'
        # 	snode[self.Atb['child_temp']][6] = snode[self.Atb['child_temp']][6] + list_to_add
        return snode

    def remove_region(self, region_label_list, snode):
        for region_label in region_label_list:
            snode[self.Atb['child_temp']][self.ChildLabel[region_label]] = []
        # if region_label == 'TL':
        # 	snode[self.Atb['child_temp']][0] = []
        # elif region_label == 'BL'
        # 	snode[self.Atb['child_temp']][1] = []
        # elif region_label == 'T'
        # 	snode[self.Atb['child_temp']][2] = []
        # elif region_label == 'B'
        # 	snode[self.Atb['child_temp']][3] = []
        # elif region_label == 'C'
        # 	snode[self.Atb['child_temp']][4] = []
        # elif region_label == 'SUP'
        # 	snode[self.Atb['child_temp']][5] = []
        # elif region_label == 'SUB'
        # 	snode[self.Atb['child_temp']][6] = []

        return snode

    def merge_region(self, region_label_list, region_label, snode):
        add_idx = self.ChildLabel[region_label]

        for region_to_merge in region_label_list:
            if region_to_merge == region_label:
                continue

            snode[self.Atb['child_temp']][add_idx] = \
                snode[self.Atb['child_temp']][add_idx] + \
                snode[self.Atb['child_temp']][self.ChildLabel[region_to_merge]]
            snode[self.Atb['child_temp']][self.ChildLabel[region_to_merge]] = []

        return snode


# obj = BBParser()
# obj.debug()
