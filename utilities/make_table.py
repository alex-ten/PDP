from tabulate import tabulate

def make_table(a, rkeys=None, ckeys=None):
    tab_list = []
    for i,k in enumerate(rkeys):
        tab_list.append([k] + list(a[i]))
    table = tabulate(tab_list,floatfmt='.2f', headers = ckeys)
    return table

