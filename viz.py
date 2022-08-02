from os.path import join as oj

import matplotlib.pyplot as plt


def savefig(fname):
    plt.savefig(oj('results', fname + '.png'), dpi=300)
    plt.savefig(oj('results', fname + '.pdf'))
