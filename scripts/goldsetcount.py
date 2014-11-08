from __future__ import print_function

import csv
import src.main

def print_psize(p):
    class_s = 0
    method_s = 0
    ids_s = 0
    for sys, d in p.items():

        if 'method' not in d:
            d['method'] = ' '
        else:
            method_s += d['method']

        if 'class' not in d:
            d['class'] = ' '
        else:
            class_s += d['class']

        ids_s += d['ids']

        print(' & '.join(map(str, [sys, d['ids'], d['class'], d['method']])), '\\\\')

    print(' & '.join(map(str, ['Total', ids_s, class_s, method_s])))




projects = src.main.load_projects()
psize = dict()
for project in projects:
    try:
        goldsets = src.main.load_goldsets(project)
    except IOError:
        continue

    size = sum(len(g) for _,g in goldsets)

    c = project.name + ' ' + project.version
    if c not in psize:
        psize[c] = dict()

    psize[c][project.level] = size

    if 'ids' not in psize[c]:
        psize[c]['ids'] = len(goldsets)

    assert psize[c]['ids'] == len(goldsets)

print_psize(psize)




p = dict()
for project in projects:
    try:
        ranks = src.main.read_ranks(project, 'temporal_lda')
    except IOError:
        continue

    ids = set([x[1] for x in ranks])
    c = project.name + ' ' + project.version
    if c not in p:
        p[c] = set()

    p[c] |= ids


psize = dict()
for project in projects:
    c = project.name + ' ' + project.version
    try:
        ids = p[c]
    except KeyError:
        continue

    goldsets = dict(src.main.load_goldsets(project))
    size = sum(len(goldsets[str(i)]) for i in ids)

    if c not in psize:
        psize[c] = dict()
    psize[c][project.level] = size
    if 'ids' not in psize[c]:
        psize[c]['ids'] = len(ids)

    assert psize[c]['ids'] == len(ids)


print_psize(psize)
