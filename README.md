Changeset Topics for Feature Location
=====================================

[![Build Status](https://magnum.travis-ci.com/cscorley/feature-location.svg?token=CjXwzG3DCkMLgDhpEr4U&branch=master)](https://magnum.travis-ci.com/cscorley/feature-location)


### Authors 

- Christopher S. Corley, <cscorley@crimson.ua.edu>
- Kelly L. Kashuda, <klkashuda@crimson.ua.edu>
- Nicholas A. Kraft, <nicholas.a.kraft@us.abb.com>


### Abstract

> Feature location is a program comprehension activity in which a developer
> inspects source code to locate the classes or methods that implement a
> feature of interest. Many feature location techniques (FLTs) are based on
> text retrieval models, and in such FLTs it is typical for the models to be
> built from source code snapshots. However, source code evolution leads to
> model obsolescence and thus to the need to retrain the model from the latest
> snapshot. In this paper, we introduce a topic-modeling-based FLT in which the
> model is built incrementally from source code *changesets*. By training an
> online learning algorithm using changesets, the FLT maintains an up-to-date
> model without incurring the non-trivial computational cost associated with
> retraining traditional FLTs. We demonstrate the robustness of the new FLT via
> an empirical study of two popular text retrieval models: latent semantic
> indexing and latent Dirichlet allocation. Overall, we studied over 1200
> defects and features from 14 open-source Java projects. We also present a
> *temporal simulation* in which we approximate how the FLT would perform
> throughout the evolution of a project.


### Description

Hello!

Most things related to this project can be found in the GitHub repository.

Some files which you might find of immediate interest:

- [Pre-print (pdf)](./paper.pdf)
- [Source code (.tar.gz)](./source.tar.gz)

### Dataset

Our dataset is available as part of the GitHub repository,
but things such as the corpora go through extraction and pre-processing steps.

**TODO: Explain the data/ layout**

All project information is in `projects.csv`, and supplementary data is under
the `data` directory:

- the repository URLs under `data/<project>/repos.txt`
- extracted release source code `data/<project>/<version>/src`

#### Corpora

**TODO: Generate corpora .gz files for download**


Basic idea: compare LDA models built with two different sources of
documents. The first being a traditional source code snapshot, and the
second being variations on a changeset snapshot.

### Code

#### Installing

Install everything using make:

    $ make install

Or, if you use virtualenv, you can `make init` instead.

Now, you should be able to run commands:

    $ cfl <project name>

e.g.,

    $ cfl mucommander

This will run the experiment on the given project name. 

To run the experiment on a certain version or level, use the `--version` and `--level` flags.

    $ cfl  mucommander --version v0.8.5 --level class

See `--help` for additional usage.

    $ cfl --help
    Usage: cfl [OPTIONS] NAME

    Changesets for Feature Location

    Options:
    --verbose
    --debug
    --path TEXT     Set the directory to work within
    --version TEXT  Version of project to run experiment on
    --level TEXT    Granularity level of project to run experiment on
    --help          Show this message and exit.
    
