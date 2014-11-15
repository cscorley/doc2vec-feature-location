
from __future__ import print_function

import csv
import src.main
import src.utils
import logging
import sys
import scipy.stats

def get_p(p):
    if p < 0.01:
        return "p < 0.01"
    if p >= 0.01:
        return "p = %f" % p

    if p < 0.05:
        return "p < 0.05"
    if p < 0.1:
        return "p < 0.1"

def ap(project, t):
    ranks = src.main.read_ranks(project, t)
    c = project.name+project.version
    new = list()
    for r, i, g in ranks:
        new.append((r, c+str(i), g))

    return new

def print_em(desc, a, b, c, d, ignore=False):
    acc = 6
    x, y = src.main.merge_first_rels(a, c, ignore=ignore)
    T, p = scipy.stats.wilcoxon(x, y)

    lda = "%d & $%s$"  % (T, get_p(p))

    x, y = src.main.merge_first_rels(b, d, ignore=ignore)
    T, p = scipy.stats.wilcoxon(x, y)

    lsi = "%d & $%s$"  % (T, get_p(p))

    if len(c) < 20 or len(a) < 20 or len(d) < 20 or len(b) < 20:
        desc = "{\\it" + desc + "}"

    l = [desc,
         lda, lsi
        ]
    print(' & '.join(map(str, l)), '\\\\')

projects = src.main.load_projects()
r_lda = []
r_lsi = []
c_lda = []
c_lsi = []
for project in projects:
    if project.level != sys.argv[1]:
        continue

    desc = ' '.join([project.name, project.version])
    a = ap(project, 'release_lda')
    b = ap(project, 'release_lsi')

    r_lda += a
    r_lsi += b

    c = ap(project, 'changeset_lda')
    d = ap(project, 'changeset_lsi')

    c_lda += c
    c_lsi += d

    print_em(desc, a, b, c, d, ignore=False)


print('\\midrule')
print_em("All", r_lda, r_lsi, c_lda, c_lsi, ignore=False)

print()
print()
print()

r_lda = []
r_lsi = []
t_lda = []
t_lsi = []
for project in projects:
    if project.level != sys.argv[1]:
        continue

    desc = ' '.join([project.name, project.version])
    a = ap(project, 'release_lda')
    b = ap(project, 'release_lsi')

    r_lda += a
    r_lsi += b

    try:
        e = ap(project, 'temporal_lda')
        f = ap(project, 'temporal_lsi')
    except IOError:
        continue

    t_lda += e
    t_lsi += f

    print_em(desc, a, b, e, f, ignore=True)

print('\\midrule')
print_em("All", r_lda, r_lsi, t_lda, t_lsi, ignore=True)
