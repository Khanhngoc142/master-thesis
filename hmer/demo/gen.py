import os
import matplotlib.pyplot as plt
from extractor.crohme_parser import library, inkml
from utils.plt_draw import plt_draw_traces, plt_setup
from utils.fs import get_source_root, load_object, save_object
import random
import sys
from extractor.crohme_parser.fonts import StandardFont


def sample_list(lst):
    idx = random.randint(0, len(lst) - 1)
    return lst[idx]


def gen_linear_equation():
    operators = '+ -'.split()
    variables = 'x y u v'.split()
    coefficients = 'a b c A B C'.split()

    equation = [sample_list(coefficients)]
    first_var = sample_list(variables)
    equation.append(first_var)
    equation.append(sample_list(operators))
    equation.append(sample_list(coefficients))
    equation.append('=')
    cur_var = first_var
    while cur_var == first_var:
        cur_var = sample_list(variables)
    equation.append(cur_var)
    return equation


def demo_linear_equation(lib, num):
    fig, axes = plt.subplots(nrows=num)
    for i in range(num):
        eq = gen_linear_equation()
        ax = axes[i]
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        gen_eq = lib.generate_equation_traces(eq)
        plt_draw_traces([trace for trace_group in gen_eq for trace in trace_group], ax=ax)
    return fig, axes


def demo_equation_w_subsup(lib, num):
    demo_eq = ['x', '_', ['1'], '^', ['2'], '+', 'x', '_', ['2'], '^', ['3']]
    fig, axes = plt.subplots(nrows=num)
    for i in range(num):
        ax = axes[i]
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        gen_eq = lib.generate_equation_traces(demo_eq)
        plt_draw_traces([trace for trace_group in gen_eq for trace in trace_group], ax=ax)
    return fig, axes


def demo_custom_equation(lib, eq, num):
    fig, axes = plt.subplots(nrows=num)
    for i in range(num):
        ax = axes[i]
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        gen_eq = lib.generate_equation_traces(eq)
        plt_draw_traces([trace for trace_group in gen_eq for trace in trace_group], ax=ax)

    return fig, axes


if __name__ == "__main__":
    root_path = get_source_root()

    demo_lib = load_object(os.path.join(root_path, "demo-outputs", "lib.pkl"))

    # demo_linear_equation(demo_lib, 5)
    # demo_equation_w_subsup(demo_lib, 5)
    # demo_equation_w_subsup(demo_lib, 5)
    # demo_custom_equation(demo_lib, [*list('f(x)=Ax^'), ['3',], *list('+10x')], 5)
    # demo_custom_equation(demo_lib, list('f(x)=Ax+b'), 5)
    # demo_custom_equation(demo_lib, ['x', '^', ['2', '^', ['2']]], 3)
    # demo_custom_equation(demo_lib, ['x', '^', ['2', '^', ['2', '^', ['2']]]], 2)
    # demo_eq = ['\\lim', '_', ['x', '\\rightarrow'] + list('100')]
    # demo_eq = "x \\rightarrow 1 0 0".split()
    # demo_eq = ['\\sum'] + list('abc+def-ghi=jkl') + ['\\times'] + list('mnop') + ['\\rightarrow'] + list('xyz')
    # demo_eq = ['\\lim', '_', ['k', '\\rightarrow'] + list('100'), '\\sum', '_', list('x=0'), '^', ['k'], 'p', '_', ['i'], 'x', '_', ['i']]
    # demo_eq = list(StandardFont.font_metric.keys())
    # demo_eq = ['\\log', ] + list('(x)=y')
    # demo_eq = list('H(p,q)=-') + ['\\sum', '_', ['x', '\\in', 'X']] + list('p(x)') + ['\\log', '_', ['2']] + list('q(x)')
    # demo_eq = ['\\int', '_', ['-', '\\infty'], '^', ['\\infty'], *list('aibicidieifi'), '\\alpha', '\\beta', '\\gamma'] + list('abcxyz') + ['\\infty', '\\pi', *list('r_'), ['2']]
    #demo_eq = ['A', '\\prime', *list('=A')]
    #demo_eq = ['\\sqrt', list('12+x')]
    demo_eq = ['x','+','\\sqrt', ['\\sqrt', list('12+x')], *list('+y+z')]
    demo_custom_equation(demo_lib, demo_eq, 2)
    plt.show()
