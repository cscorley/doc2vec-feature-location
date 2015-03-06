
from __future__ import print_function

import csv
import src.main
import src.utils
import logging
import sys
import scipy.stats

def ap(project, t):
    goldsets = src.main.load_goldsets(project)
    ranks = src.main.read_ranks(project, t)
    frms = src.main.get_frms(goldsets, ranks)
    c = project.name+project.version
    new = list()
    for r, i, g in frms:
        new.append((r, c+str(i), g))

    return new


def print_em(desc, a, b, ignore=False, file=None):
    acc = 6
    x, y = src.main.merge_first_rels(b, a, ignore=ignore)
    T, p = scipy.stats.wilcoxon(x, y, correction=True)


    changeset = round(src.utils.calculate_mrr(x), acc)
    snapshot = round(src.utils.calculate_mrr(y), acc)
    if changeset >= snapshot:
        changeset = "{\\bf %f }" % changeset
        snapshot = "%f" % snapshot
    else:
        snapshot = "{\\bf %f }" % snapshot
        changeset = "%f" % changeset

    if len(x) < 10:
        star = "^{*}"
    else:
        star = ''

    if p < 0.01:
        p = "$p < 0.01%s$" % star
    else:
        p = "$p = %f%s$" % (p, star)


    l = [desc,
        snapshot, changeset,
        p
        ]

    print(' & '.join(map(str, l)), '\\\\', file=file)

HEADER="""\\begin{table}[t]
\\renewcommand{\\arraystretch}{1.3}
\\footnotesize
\\centering"""
INNER_HEADER="""\\caption{{\\bf %s}: MRR and $p$-values of %s-level %s}
\\begin{tabular}{l|ll|ll}
\\toprule
Subject System & %s & %s & $p$-value  \\\\
\\midrule"""
INNER_FOOTER= "\\bottomrule\n\\end{tabular}\n\\label{table:%s:%s:%s}"
FOOTER="\\end{table}"

projects = src.main.load_projects()
rq2_projects = ['argouml', 'jabref', 'jedit', 'mucommander']

for kind in ['lda']: # 'lsi']:
    alldict = dict()
    with open('paper/tables/rq1_%s.tex' % kind, 'w') as f:
        print(HEADER, file=f)
        for level in ['class', 'method']:
            rname = 'release' # + kind
            cname = 'changeset' # + kind
            alldict[rname] = list()
            alldict[cname] = list()
            print(INNER_HEADER % ('RQ1', level, 'Batch', 'Snapshot', 'Changeset'), file=f)
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
            print(INNER_FOOTER % ('rq1', level, kind), file=f)

        print(FOOTER, file=f)

    alldict = dict()
    with open('paper/tables/rq2_%s.tex' % kind, 'w') as f:
        print(HEADER, file=f)
        for level in ['class', 'method']:
            rname = 'changeset' # + kind
            cname = 'temporal' #+ kind
            alldict[rname] = list()
            alldict[cname] = list()
            print(INNER_HEADER % ('RQ2', level, 'Temporal', 'Batch', 'Temporal'), file=f)
            for project in projects:
                if project.level != level:
                    continue
                if project.name not in rq2_projects:
                    continue

                desc = ' '.join([project.printable_name, project.version])

                a = ap(project, rname)
                try:
                    b = ap(project, cname)
                except IOError:
                    continue # some projects don't have temporal junk

                alldict[rname] += a
                alldict[cname] += b

                print_em(desc, a, b, ignore=True, file=f)

            print('\\midrule', file=f)
            print_em("All", alldict[rname], alldict[cname], ignore=True, file=f)
            print(INNER_FOOTER % ('rq2', level, kind), file=f)
        print(FOOTER, file=f)
