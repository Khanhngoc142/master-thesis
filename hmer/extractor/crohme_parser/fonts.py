class StandardFont:
    font_metric = {  # font metric: symbols -> (standard_scale, standard_y_shift)
        '!': (1, 0),
        '(': (1, 0),
        ')': (1, 0),
        '[': (1, 0),
        ']': (1, 0),
        '\\{': (1, 0),
        '\\}': (1, 0),
        '|': (1, 0),
        '+': (0.6, 0.65),  # (0.7, 0.35)
        '-': (0.6, 0.65),  # (0.7, 0.45)
        '\\times': (0.6, 0.5),
        '\\div': (0.6, 0.5),
        '/': (1, 0),
        '=': (0.6, 0.65),  # (0.7, 0.4)
        '\\rightarrow': (1, 0.45),
        ',': (0.1, 0.8),
        '.': (0.1, 0.8),
        '0': (1, 0),
        '1': (1, 0),
        '2': (1, 0),
        '3': (1, 0),
        '4': (1, 0),
        '5': (1, 0),
        '6': (1, 0),
        '7': (1, 0),
        '8': (1, 0),
        '9': (1, 0),
        'A': (1, 0),
        'B': (1, 0),
        'C': (1, 0),
        'E': (1, 0),
        'F': (1, 0),
        'G': (1, 0),
        'H': (1, 0),
        'I': (1, 0),
        'L': (1, 0),
        'M': (1, 0),
        'N': (1, 0),
        'P': (1, 0),
        'R': (1, 0),
        'S': (1, 0),
        'T': (1, 0),
        'V': (1, 0),
        'X': (1, 0),
        'Y': (1, 0),
        'a': (0.6, 0.4),
        'b': (1, 0),
        'c': (0.6, 0.4),
        'd': (1, 0),
        'e': (0.6, 0.4),
        'f': (1.2, 0),
        'g': (1.2, 0.4),
        'h': (1, 0),
        'i': (0.7, 0.3),
        'j': (1.4, 0.4),
        'k': (1, 0),
        'l': (1, 0),
        'm': (0.6, 0.4),
        'n': (0.6, 0.4),
        'o': (0.6, 0.4),
        'p': (1.2, 0.4),
        'q': (1.2, 0.4),
        'r': (0.6, 0.4),
        's': (0.6, 0.4),
        't': (0.8, 0.2),
        'u': (0.6, 0.4),
        'v': (0.6, 0.4),
        'w': (0.6, 0.4),
        'x': (0.6, 0.4),
        'y': (1.2, 0.4),
        'z': (0.6, 0.4),

        '\\lim': (1, 0),
        '\\cos': (0.6, 0.4),
        '\\sin': (0.7, 0.3),
        '\\tan': (1, 0),
        '\\theta': (0.95, 0.05),

        '\\sum': (1.2, -0.1),
        '\\exists': (1.2, -0.1),
        '\\forall': (1.2, -0.1),

        '\\geq': (1, 0.5),
        '\\gt': (1, 0.5),
        '\\leq': (1, 0.5),
        '\\lt': (1, 0.5),
        '\\neq': (1, 0.5),

        '\\in': (0.8, 0.2),

        '\\Delta': (1, 0),

        # '\\alpha': (1, 0),
        # '\\beta': (1, 0),
        # '\\gamma': (1, 0),
        # '\\infty': (1, 0),
        # '\\int': (1, 0),
        # '\\lambda': (1, 0),
        # '\\ldots': (1, 0),

        # '\\log': (,),
        # '\\mu': (,),
        # '\\phi': (,),
        # '\\pi': (,),
        # '\\pm': (,),
        # '\\prime': (,),
        # '\\sigma': (,),
        # '\\sqrt': (,),

    }
    supsub_equation_scale = 0.5
    lowerupper_equation_scale = 0.4
    sub_equation_yshift = 0.7
    sup_equation_yshift = 0.3
    upper_equation_yshift = lower_equation_yshift = 0.2
    symbol_gap = 0.2

    @staticmethod
    def upperlower_xshift_rightalign(child_w, parent_w):
        return child_w - parent_w

    @staticmethod
    def upperlower_xshift_leftalign(child_w, parent_w):
        return 0

    @staticmethod
    def upperlower_xshift_center(child_w, parent_w):
        return (child_w - parent_w) / 2
