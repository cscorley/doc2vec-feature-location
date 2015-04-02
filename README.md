Modeling Changeset Topics for Feature Location
==============================================

### Authors

- Christopher S. Corley, <cscorley@crimson.ua.edu>
- Kelly L. Kashuda, <klkashuda@crimson.ua.edu>
- Nicholas A. Kraft, <nicholas.a.kraft@us.abb.com>

### Abstract

>Feature location is a program comprehension activity in which a developer
>inspects source code to locate the classes or methods that implement a feature
>of interest. Many feature location techniques (FLTs) are based on text
>retrieval models, and in such FLTs it is typical for the models to be trained
>on source code snapshots. However, source code evolution leads to model
>obsolescence and thus to the need to retrain the model from the latest
>snapshot. In this paper, we introduce a topic-modeling-based FLT in which the
>model is built incrementally from source code history. By training an online
>learning algorithm using changesets, the FLT maintains an up-to-date model
>without incurring the non-trivial computational cost associated with
>retraining traditional FLTs. Overall, we studied over 1,200 defects and
>features from 14 open-source Java projects. We also present a historical
>simulation that demonstrates how the FLT performs as a project evolves. Our
>results indicate that the accuracy of a changeset-based FLT is similar to that
>of a snapshot-based FLT, but without the retraining costs.


### Description

Hello!

Most things related to this project can be found in the GitHub repository.

Some files which you might find of immediate interest:

- [Pre-print (pdf)](./paper.pdf)
- [Release (.tar.gz)](./release.tar.gz) -- Includes source code and starter data
- [Full dataset (.7z)](./fulldata.7z) (6.8GB) -- The full dataset, including
  generated corpora, models, and results. 
- [Taser](https://github.com/nkraft/taser) -- Tool used for parsing Java source
- [GitHub Repository]() -- tbd

We include all of these sources (even the embarrassing commit history) to help
encourage others in software engineering to engage in
[open science principles](http://en.wikipedia.org/wiki/Open_Science).
If there is anything you'd like to see included, feel free to email at
<cscorley@crimson.ua.edu>. ***Please*** don't download the full dataset unless
you're absolutely sure you need it; I have limited bandwidth :)

### Dataset

Our dataset is available as part of the GitHub repository (and release archive),
but things such as the corpora go through extraction and pre-processing steps.

All project information is in `projects.csv`, and supplementary data is under
the `data/` directory.

#### `data/` layout

Each subdirectory of `data/` follows the following schema:

- `<project>/` -- the main project name (from `projects.csv`)
    - `repos.txt` -- the repository URLs for cloning
    - `svn2git.csv` -- if the repository was converted to git from
    subversion, this is the SVN revision -> Git sha mapping
    - `<version>/` -- version specific files:
        - `ids.txt` -- line separated query ids that relate to the issue
          report ids or feature request ids used in the datasets.
        - `queries/` -- contains the *unpreprocessed* queries:
            - `ShortDescription<id>.txt` -- the title or summary of the query
            - `LongDescription<id>.txt` -- the full description of the query
        - `goldsets/` -- contains the goldset files:
            - `class/` -- contains class-level goldsets
                - `<id>.txt` -- line separated class names related to the query id
            - `method/` -- contains method-level goldsets
                - `<id>.txt` -- line separated method names related to the query id
        - `<FLT>_<topic modeler>-<level>-ranks.csv` -- the effectiveness
          measures for each of the various experiment setups, e.g.,
          `changeset_lda-class-ranks.csv` is the ranks of the batch-mode LDA
          experiment at the class-level.

There may be other files in the directories which originate from the two
original datasets.


### Code

We've done our best to include every possible bit of code written in order to
complete this paper. Much of the main experiment is under `src/`, with a couple
of corresponding tests in `tests/`, and some helper scripts in `scripts` that
helped to convert a dataset to our format or generate tables for the paper.

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
