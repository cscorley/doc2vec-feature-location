Doc2vec for Feature Location
==============================================

### Authors

- Christopher S. Corley, <cscorley@crimson.ua.edu>
- Nicholas A. Kraft, <nicholas.a.kraft@us.abb.com>

### Abstract

> hi

### Description

Hello!

Most things related to this project can be found in the GitHub repository.

Some files which you might find of immediate interest:


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

    $ dfl <project name>

e.g.,

    $ dfl mucommander

This will run the experiment on the given project name.

To run the experiment on a certain version or level, use the `--version` and `--level` flags.

    $ dfl  mucommander --version v0.8.5 --level class

See `--help` for additional usage.

    $ dfl --help
    Usage: dfl [OPTIONS] NAME

    Changesets for Feature Location

    Options:
    --verbose
    --debug
    --path TEXT     Set the directory to work within
    --version TEXT  Version of project to run experiment on
    --level TEXT    Granularity level of project to run experiment on
    --help          Show this message and exit.
