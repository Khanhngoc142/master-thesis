import numpy as np
import pandas as pd
import random
from utilities.fs import get_source_root, load_object
import os
import string

root_path = get_source_root()
# demo_lib = load_object(os.path.join(root_path, "demo-outputs", "lib.pkl"))
pd_symbol_weight = pd.read_csv(os.path.join(root_path, "metadata", "training_symbol_weight.csv"))

lowercase_alphabet_sym = list(string.ascii_lowercase)
uppercase_alphabet_sym = list(set(list(string.ascii_uppercase)) - set(['D', 'J', 'K', 'O', 'Q', 'U', 'W', 'Z']))
digit_sym = list(string.digits)
digit_without_zero_sym = list(string.digits.replace('0', ''))
latin_sym = '\\theta \\Delta \\alpha \\beta \\gamma \\lambda \\mu \\phi'.split(' ')

operator_sym = '+ - \\times \\div /'.split(' ')
compare_operator_sym = '\\lt \\gt \\leq \\geq \\neq'.split(' ')
pair_sym = ['\\[', '\\{']


def get_symbol_weight(sym_lst):
    return [pd_symbol_weight[pd_symbol_weight['symbol_name'] == c].weight.item() for c in sym_lst]


def random_choices(candidates, random_k):
    return random.choices(candidates, weights=get_symbol_weight(candidates), k=random_k)


def gen_number():
    eq = []
    first_digit = random_choices(digit_without_zero_sym, 1)
    eq.extend(first_digit)

    random_k = random.randint(0, 2)
    continuous_digit = random_choices(digit_sym, random_k)
    eq.extend(continuous_digit)

    random_type = random.randint(1,3)
    if random_type == 1:
        eq.append('\\pi')
    elif random_type == 2:
        eq.append('!')

    return eq


def gen_linear_operand():
    random_type = random.randint(1, 3)
    if random_type == 1:
        letters = []
        letters.extend(random_choices(uppercase_alphabet_sym + lowercase_alphabet_sym + latin_sym, 1))
        extra_random = random.randint(1, 2)
        if extra_random == 1:
            letters.append('\\prime')
        return letters
    else:
        eq = []
        eq.extend(gen_number())

        if random_type == 2:
            return eq
        else:
            last_letter = random_choices(uppercase_alphabet_sym + lowercase_alphabet_sym + latin_sym, 1)
            eq.extend(last_letter)
            return eq


def gen_basic_linear():
    eq = []
    operand_1 = gen_linear_operand()
    eq.extend(operand_1)
    operator = random_choices(operator_sym, 1)
    eq.extend(operator)
    operand_2 = gen_linear_operand()
    eq.extend(operand_2)

    return eq


def gen_frac():
    eq = ['f', '=', '\\frac']
    numerator = gen_basic_linear()
    denominator = gen_linear_operand()

    r = random.randint(1, 2)
    if r == 1:
        eq.append(numerator)
        eq.append(denominator)
    else:
        eq.append(denominator)
        eq.append(numerator)

    return eq


def gen_log():
    eq = ['\\log', '_']

    base = []
    first_digit = random_choices(digit_without_zero_sym, 1)

    base.extend(first_digit)

    random_k = random.randint(0, 1)
    continuous_digit = random_choices(digit_sym, random_k)
    base.extend(continuous_digit)

    eq.append(base)
    main = gen_linear_operand()
    eq.extend(main)

    return eq


def gen_sin_cos():
    eq = []
    sin_or_cos = random_choices(['\\sin', '\\cos'], 1)
    eq.extend(sin_or_cos)

    base = gen_linear_operand()
    eq.extend(base)

    return eq


def gen_condition():
    eq = []
    forall_or_exists = random_choices(['\\forall', '\\exists'], 1)
    eq.extend(forall_or_exists)

    variable = random_choices(lowercase_alphabet_sym, 1)
    eq.extend(variable)
    eq.append('\\in')
    eq.append('[')

    lower_bound = random_choices(digit_sym, 1)
    upper_bound = ['1']
    while int(upper_bound[0]) <= int(lower_bound[0]):
        upper_bound = random_choices(digit_sym, 1)

    eq.extend(lower_bound)
    eq.append(',')
    eq.extend(upper_bound)
    eq.append(']')

    eq.extend(variable)
    compare_operator = random_choices(compare_operator_sym, 1)
    number = gen_number()

    eq.extend(compare_operator)
    eq.extend(number)

    return eq


def gen_unary():
    eq = []

    or_gt = random_choices(['!', '\\pm'], 1)
    number = gen_number()
    # eq.extend(or_gt)
    eq.extend(number)

    if or_gt[0] == '!':
        eq.extend(or_gt)
    else:
        eq.insert(0, or_gt[0])

    return eq


def gen_sqrt():
    eq = ['\\sqrt']
    base = gen_basic_linear()
    eq.append(base)

    return eq


def gen_lim():
    eq = ['\\lim', '_']

    sub = []
    variable = random_choices(latin_sym, 1)
    number = gen_number()
    sub.extend(variable)
    sub.append('\\rightarrow')
    sub.extend(number)

    eq.append(sub)

    base = []
    operator = random_choices(operator_sym, 1)
    number =gen_number()
    base.extend(variable)
    base.extend(operator)
    base.extend(number)

    eq.extend(base)

    return eq


def gen_other():
    eq = ['\\{']
    variable = random_choices(latin_sym + uppercase_alphabet_sym, 1)
    number = gen_number()
    eq.extend(variable)
    random_type = random.randint(1,2)
    if random_type == 1:
        eq.append('\\rightarrow')
    else:
        eq.append('\\ldots')
    eq.extend(number)
    eq.append('\\}')

    return eq


def gen_selection(random_type):
    # random_type = random.randint(1, 10)
    if random_type == 1:
        return gen_number()
    elif random_type == 2:
        return gen_linear_operand()
    elif random_type == 3:
        return gen_basic_linear()
    elif random_type == 4:
        return gen_sin_cos()
    elif random_type == 5:
        return gen_frac()
    elif random_type == 6:
        return gen_lim()
    elif random_type == 7:
        return gen_log()
    elif random_type == 8:
        return gen_unary()
    elif random_type == 9:
        return gen_condition()
    elif random_type == 10:
        return gen_other()


def gen_mix():
    combination_or_not = random.randint(1,2)
    if combination_or_not == 1:
        model_type = random.randint(1, 10)
        return gen_selection(model_type)
    else:
        eq = []
        operand_1 = gen_selection(random.randint(1,7))
        operand_2 = gen_selection(random.randint(1, 7))
        operator = random_choices(operator_sym, 1)

        eq.extend(operand_1)
        eq.extend(operator)
        eq.extend(operand_2)
        return eq




