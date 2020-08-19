import os
import matplotlib.pyplot as plt
from extractor.crohme_parser import library, inkml
from utils.plt_draw import plt_draw_traces, plt_setup
from utils.fs import get_source_root, load_object, save_object
import random


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


if __name__ == "__main__":
    root_path = get_source_root()

    demo_lib = load_object(os.path.join(root_path, "demo-outputs", "lib.pkl"))

    # update font metrics
    dummy_lib = library.Library()
    demo_lib._font_metric = dummy_lib._font_metric

    demo_linear_equation(demo_lib, 5)
    plt.show()
