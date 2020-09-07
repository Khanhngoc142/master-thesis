
symbols = [
    '!',
    '(',
    ')',
    '[',
    ']',
    '\\{',
    '\\}',
    '|',
    '+',
    '-',
    '\\times',
    '\\div',
    '/',
    '=',
    '\\rightarrow',
    ',',
    '.',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'A',
    'B',
    'C',
    'E',
    'F',
    'G',
    'H',
    'I',
    'L',
    'M',
    'N',
    'P',
    'R',
    'S',
    'T',
    'V',
    'X',
    'Y',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
    '\\infty',
    '\\pi',
    '\\cos',
    '\\sin',
    '\\tan',
    '\\theta',
    '\\lim',
    '\\sum',
    '\\exists',
    '\\forall',
    '\\geq',
    '\\gt',
    '\\leq',
    '\\lt',
    '\\neq',
    '\\in',
    '\\Delta',
    '\\log',
    '\\alpha',
    '\\beta',
    '\\gamma',
    '\\int',
    '\\ldots',
    '\\lambda',
    '\\mu',
    '\\phi',
    '\\pm',
    '\\prime',
    '\\sigma',
    '\\sqrt',
]

symbol2idx = dict([(s, idx + 1) for idx, s in enumerate(symbols)])


def normalize_label(label):
    """
    normalize label from a nested list form to flatten list form and lookup index
    :param label: nested list
    :return:
    """
    for s in label:
        if isinstance(s, list):
            for out_s in normalize_label(s):
                yield symbol2idx[out_s]
        elif isinstance(s, str):
            yield symbol2idx[s]
