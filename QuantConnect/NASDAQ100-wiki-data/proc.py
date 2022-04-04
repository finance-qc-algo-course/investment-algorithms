#! /usr/bin/python3

import re

PATTERN="\(\w+\)"

if __name__ == '__main__':
    components = None
    for year in range(2006, 2015):
        fname = "{}.txt".format(year)
        cur_components = set()
        with open(fname, 'r') as file:
            data = file.read()
            cur_components = set(comp[1 : -1] \
                    for comp in re.findall(PATTERN, data))

        components = \
                cur_components if components is None \
                else components.intersection(cur_components)

    print([comp for comp in components])

