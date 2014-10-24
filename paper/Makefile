PAPER 		= paper

LATEX 		= pdflatex
BIBTEX		= bibtex

GRAPHICS	= graphics

EPS_GFX		= $(shell echo $(GRAPHICS)/*.eps)
GEN_GFX 	= $(addsuffix .pdf, $(basename $(EPS_GFX)))
GFX_FILES	= #$(GEN_GFX)

CLS_FILES	= $(shell find . -name '*.cls')
TEX_FILES	= $(shell find . -name '*.tex')
BIB_FILES	= $(shell find . -name '*.bib')

DEP_FILES	= $(CLS_FILES) $(TEX_FILES) $(BIB_FILES) $(GFX_FILES)


all: $(PAPER).pdf

$(PAPER).pdf: $(DEP_FILES)
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
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

edit:
	$(EDITOR) paper.tex sec_intro.tex sec_related.tex sec_methodology.tex sec_study.tex sec_threats.tex sec_conclusion.tex

%.pdf: %.eps
	epstopdf --outfile=$@ $<

OS = $(shell uname -s)
ifeq ($(strip $(OS)),Linux)
	PDF_VIEW = xdg-open
else
	PDF_VIEW = open -a /Applications/Preview.app
endif

view: $(PAPER).pdf
	$(PDF_VIEW) $(PAPER).pdf &
