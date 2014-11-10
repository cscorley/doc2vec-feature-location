
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import gensim
import src.main


# In[2]:

def get_all_ranks(project):
    r_lda = [x for x,y,z in src.main.read_ranks(project, 'release_lda')]
    r_lsi = [x for x,y,z in src.main.read_ranks(project, 'release_lsi')]
    c_lda = [x for x,y,z in src.main.read_ranks(project, 'changeset_lda')]
    c_lsi = [x for x,y,z in src.main.read_ranks(project, 'changeset_lsi')]
    try:
        t_lda = [x for x,y,z in src.main.read_ranks(project, 'temporal_lda')]
        t_lsi = [x for x,y,z in src.main.read_ranks(project, 'temporal_lsi')]
    except:
        t_lda = []
        t_lsi = []

    return r_lda, c_lda, t_lda, r_lsi, c_lsi, t_lsi


# In[3]:

projects = src.main.load_projects()


# In[8]:

for project in projects:
    ranks = get_all_ranks(project)
    fig = plt.figure(dpi=300)
    fig.gca().boxplot(ranks,
                      labels=['S-LDA', 'C-LDA', 'T-LDA', 'S-LSI', 'C-LSI', 'T-LSI'])
    fig.gca().set_title(' '.join([project.name, project.version, project.level]))
    plt.savefig('paper/figures/' + project.name + project.version + project.level + '.png')
    plt.close()

# In[ ]:



