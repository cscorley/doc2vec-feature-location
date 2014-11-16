
from __future__ import print_function

import csv
import src.main
import src.utils
import logging
import sys
import scipy.stats

def ap(project, t):
    ranks = src.main.read_ranks(project, t)
    c = project.name+project.version
    new = list()
    for r, i, g in ranks:
        new.append((r, c+str(i), g))

    return new


def print_em(desc, a, b, ignore=False, file=None):
    acc = 6
    x, y = src.main.merge_first_rels(b, a, ignore=ignore)
    T, p = scipy.stats.wilcoxon(x, y)


    changeset = round(src.utils.calculate_mrr(x), acc)
    snapshot = round(src.utils.calculate_mrr(y), acc)
    if changeset >= snapshot:
        changeset = "{\\bf %f }" % changeset
        snapshot = "%f" % snapshot
    else:
        snapshot = "{\\bf %f }" % snapshot
        changeset = "%f" % changeset

    if p < 0.01:
        p = "$p < 0.01$"
    else:
        p = "$p = %f$" % p

    l = [desc,
        snapshot, changeset,
        p
        ]

    print(' & '.join(map(str, l)), '\\\\', file=file)

HEADER="""
\\begin{table}[t]
\\renewcommand{\\arraystretch}{1.3}
\\footnotesize
\\centering
\\caption{%s: MRR and Wilcoxon signed-ranksum for %s %s}
\\begin{tabular}{l|ll|ll}
   \\toprule
    Subject System & %s & %s & p-value  \\\\
    \\midrule
"""
FOOTER="""
    \\bottomrule
\\end{tabular}
\\label{table:%s:%s:%s}
\\end{table}
"""

projects = src.main.load_projects()

for level in ['class', 'method']:
    alldict = dict()
    for kind in ['lda', 'lsi']:
        rname = 'release_' + kind
        cname = 'changeset_' + kind
        alldict[rname] = list()
        alldict[cname] = list()
        with open('paper/tables/rq1_%s_%s.tex' % (level, kind), 'w') as f:
            print(HEADER % ('RQ1', level, kind.upper(), 'Snapshot', 'Changeset'), file=f)
            for project in projects:
                if project.level != level:
                    continue

                desc = ' '.join([project.printable_name, project.version])

                a = ap(project, rname)
                b = ap(project, cname)

                alldict[rname] += a
                alldict[cname] += b

                print_em(desc, a, b, ignore=False, file=f)

            print('\\midrule', file=f)
            print_em("All", alldict[rname], alldict[cname], ignore=False, file=f)
            print(FOOTER % ('rq1', level, kind), file=f)

    alldict = dict()
    for kind in ['lda', 'lsi']:
        rname = 'release_' + kind
        cname = 'temporal_' + kind
        alldict[rname] = list()
        alldict[cname] = list()
        with open('paper/tables/rq2_%s_%s.tex' % (level, kind), 'w') as f:
            print(HEADER % ('RQ2', level, kind.upper(), 'Snapshot', 'Changeset'), file=f)
            for project in projects:
                if project.level != level:
                    continue

                desc = ' '.join([project.printable_name, project.version])

                a = ap(project, rname)
                try:
                    b = ap(project, cname)
                except IOError:
                    continue # some projects don't have temporal junk

                alldict[rname] += a
                alldict[cname] += b

                print_em(desc, a, b, ignore=False, file=f)

            print('\\midrule', file=f)
            print_em("All", alldict[rname], alldict[cname], ignore=False, file=f)
            print(FOOTER % ('rq2', level, kind), file=f)
