
from __future__ import print_function

import csv
import src.main
import src.utils
import logging
import sys

def ap(project, t):
    ranks = src.main.read_ranks(project, t)
    c = project.name+project.version
    new = list()
    for r, i, g in ranks:
        new.append((r, c+str(i), g))

    return new

def print_em(desc, a, b, c, d, e, f):
    acc = 6
    x, y = src.main.merge_first_rels(c, a)
    changeset_lda = round(src.utils.calculate_mrr(x), acc)
    snapshot_lda = round(src.utils.calculate_mrr(y), acc)
    if changeset_lda > snapshot_lda:
        changeset_lda = "{\\bf %f }" % changeset_lda
        snapshot_lda = "%f" % snapshot_lda
    elif changeset_lda < snapshot_lda:
        snapshot_lda = "{\\bf %f }" % snapshot_lda
        changeset_lda = "%f" % changeset_lda

    x, y = src.main.merge_first_rels(d, b)
    changeset_lsi = round(src.utils.calculate_mrr(x), acc)
    snapshot_lsi = round(src.utils.calculate_mrr(y), acc)
    if changeset_lsi > snapshot_lsi:
        changeset_lsi = "{\\bf %f }" % changeset_lsi
        snapshot_lsi = "%f" % snapshot_lsi
    elif changeset_lsi < snapshot_lsi:
        snapshot_lsi = "{\\bf %f }" % snapshot_lsi
        changeset_lsi = "%f" % changeset_lsi

    x, y = src.main.merge_first_rels(e, a, ignore=True)
    t_changeset_lda = round(src.utils.calculate_mrr(x), acc)
    t_snapshot_lda = round(src.utils.calculate_mrr(y), acc)
    if t_changeset_lda > t_snapshot_lda:
        t_changeset_lda = "{\\bf %f }" % t_changeset_lda
        t_snapshot_lda = "%f" % t_snapshot_lda
    elif t_changeset_lda < t_snapshot_lda:
        t_snapshot_lda = "{\\bf %f }" % t_snapshot_lda
        t_changeset_lda = "%f" % t_changeset_lda

    x, y = src.main.merge_first_rels(f, b, ignore=True)
    t_changeset_lsi = round(src.utils.calculate_mrr(x), acc)
    t_snapshot_lsi = round(src.utils.calculate_mrr(y), acc)
    if t_changeset_lsi > t_snapshot_lsi:
        t_changeset_lsi = "{\\bf %f }" % t_changeset_lsi
        t_snapshot_lsi = "%f" % t_snapshot_lsi
    elif t_changeset_lsi < t_snapshot_lsi:
        t_snapshot_lsi = "{\\bf %f }" % t_snapshot_lsi
        t_changeset_lsi = "%f" % t_changeset_lsi

    l = [desc,
        snapshot_lda, changeset_lda,
        snapshot_lsi, changeset_lsi,
        t_snapshot_lda, t_changeset_lda,
        t_snapshot_lsi, t_changeset_lsi,
        ]

    print(' & '.join(map(str, l)), '\\\\')

projects = src.main.load_projects()
t_lda = []
t_lsi = []
r_lda = []
r_lsi = []
c_lda = []
c_lsi = []
for project in projects:
    if project.level != sys.argv[1]:
        continue

    desc = ' '.join([project.name, project.version])
    try:
        a = ap(project, 'release_lda')
        b = ap(project, 'release_lsi')
    except IOError:
        a = []
        b = []
        print('MISSING RELEASE', project.name, project.version, project.level)

    r_lda += a
    r_lsi += b

    try:
        c = ap(project, 'changeset_lda')
        d = ap(project, 'changeset_lsi')
    except IOError:
        c = []
        d = []
        print('MISSING CHANGESET', project.name, project.version, project.level)

    c_lda += c
    c_lsi += d

    try:
        e = ap(project, 'temporal_lda')
        f = ap(project, 'temporal_lsi')
    except IOError:
        e = []
        f = []
        print('MISSING TEMPORAL', project.name, project.version, project.level)

    t_lda += e
    t_lsi += f

    print_em(desc, a, b, c, d, e, f)


print_em("ALL", r_lda, r_lsi, c_lda, c_lsi, t_lda, t_lsi)
