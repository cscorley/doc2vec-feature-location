
from __future__ import print_function

import csv
import src.main
import logging

def ap(project, t):
    ranks = src.main.read_ranks(project, t)
    c = project.name+project.version
    new = list()
    for r, i, g in ranks:
        new.append((r, c+str(i), g))

    return new

projects = src.main.load_projects()
t_lda = []
t_lsi = []
r_lda = []
r_lsi = []
c_lda = []
c_lsi = []
for project in projects:
    try:
        r_lda += ap(project, 'release_lda')
        r_lsi += ap(project, 'release_lsi')
    except IOError:
        print('MISSING RELEASE', project.name, project.version, project.level)
    try:
        c_lda += ap(project, 'changeset_lda')
        c_lsi += ap(project, 'changeset_lsi')
    except IOError:
        print('MISSING CHANGESET', project.name, project.version, project.level)
    try:
        t_lda += ap(project, 'temporal_lda')
        t_lsi += ap(project, 'temporal_lsi')
    except IOError:
        print('MISSING TEMPORAL', project.name, project.version, project.level)

print("ALL")
src.main.do_science('basic_lda', c_lda, r_lda)
src.main.do_science('basic_lsi', c_lsi, r_lsi)

src.main.do_science('temporal_lda', t_lda, r_lda, ignore=True)
src.main.do_science('temporal_lsi', t_lsi, r_lsi, ignore=True)
