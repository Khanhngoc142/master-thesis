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

    random_type = random.randint(1,2)
    if random_type == 1:
        eq.append('\\pi')

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

    print(eq)
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

    print(eq)
    return eq


def gen_sin_cos():
    eq = []
    sin_or_cos = random_choices(['\\sin', '\\cos'], 1)
    eq.extend(sin_or_cos)

    base = gen_linear_operand()
    eq.extend(base)

    print(eq)
    return eq
