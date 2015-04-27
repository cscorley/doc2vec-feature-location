PAPER     = paper
LATEX     = pdflatex -halt-on-error
BIBTEX    = bibtex

CLS_FILES = $(shell find . -name '*.cls')
TEX_FILES = $(shell find . -name '*.tex')
BIB_FILES = $(shell find . -name '*.bib')
GFX_FILES = $(shell find ./figures)

DEP_FILES = $(CLS_FILES) $(TEX_FILES) $(BIB_FILES) $(GFX_FILES)

all: $(PAPER).pdf

.ONESHELL edit:
	make view &> /dev/null
	while true; do \
		sleep 1 && \
		inotifywait -r -e modify,attrib,close_write,move,create,delete . && \
		make; done &> /dev/null &
	$(EDITOR) $(TEX_FILES) && kill %1

$(PAPER).pdf: $(DEP_FILES)
	$(LATEX) $(PAPER) || $(BIBTEX) $(PAPER)
	$(BIBTEX) $(PAPER) || $(LATEX) $(PAPER)
	$(LATEX) $(PAPER) || $(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)

tidy:
	$(RM) *.aux
	$(RM) *.log
	$(RM) *.out
	$(RM) *.toc
	$(RM) *.bbl
	$(RM) *.blg

clean: tidy
	$(RM) $(PAPER).pdf

OS = $(shell uname -s)
ifeq ($(strip $(OS)),Linux)
	PDF_VIEW = xdg-open
else
	PDF_VIEW = open -a /Applications/Preview.app
endif

view: $(PAPER).pdf
	$(PDF_VIEW) $(PAPER).pdf &
