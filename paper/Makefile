# Paper build. Copied here by new-project.sh; edit freely.
#
#   make            compile, then delete every artifact except the PDF
#   make figures    export paper/figures/src/*.drawio to PDF
#   make check      verify figure font sizes against caption_pt
#   make lint       house prose lint over sections/
#   make diff       marked-up PDF of what changed (REV=HEAD~1)
#   make watch      recompile on save
#   make clean      remove artifacts, keep the PDF
#   make distclean  remove the PDF too
#
# The default target always cleans up after itself. Artifacts never accumulate
# next to main.tex.

MAIN     := main
TEX      := pdflatex -interaction=nonstopmode -halt-on-error -file-line-error
SKILLS   := $(HOME)/.claude/skills
SKILL    := $(or $(CLAUDE_SKILL_DIR),$(SKILLS)/research)
META     := .paper-meta.yml

# Everything LaTeX and friends leave behind. main.pdf is never in this list.
JUNK := *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz \
        *.bbl *.blg *.brf *.bcf *.run.xml *.nav *.snm *.vrb *.idx *.ilg *.ind \
        *.spl *.xdv *.thm *.loa *.los sections/*.aux

.PHONY: all pdf clean distclean figures check lint diff watch help

all: pdf

pdf: $(MAIN).pdf

# Full compile: pass, bibliography, two more passes to settle references.
# Then clean, so the directory holds main.tex and main.pdf and nothing else.
$(MAIN).pdf: $(MAIN).tex $(wildcard sections/*.tex) $(wildcard tables/*.tex) \
             $(wildcard figures/*.pdf) refs.bib results.tex
	@echo "==> pass 1"
	@$(TEX) $(MAIN).tex >/dev/null || { $(TEX) $(MAIN).tex | tail -30; exit 1; }
	@if [ -f $(MAIN).bcf ]; then \
	   echo "==> biber"; biber $(MAIN) >/dev/null 2>&1 || true; \
	 elif grep -q '\\bibliography{' $(MAIN).tex 2>/dev/null || [ -f $(MAIN).aux ]; then \
	   echo "==> bibtex"; bibtex $(MAIN) >/dev/null 2>&1 || true; \
	 fi
	@echo "==> pass 2"
	@$(TEX) $(MAIN).tex >/dev/null 2>&1 || true
	@echo "==> pass 3"
	@$(TEX) $(MAIN).tex >/dev/null 2>&1 || true
	@$(MAKE) --no-print-directory warnings
	@$(MAKE) --no-print-directory clean
	@echo "==> $(MAIN).pdf ($$(du -h $(MAIN).pdf | cut -f1)), artifacts removed"

# Surface the warnings that matter before the log is deleted.
warnings:
	@if [ -f $(MAIN).log ]; then \
	  grep -c 'Citation.*undefined' $(MAIN).log 2>/dev/null | grep -qv '^0$$' \
	    && echo "    WARNING: undefined citations" || true; \
	  grep -c 'Reference.*undefined' $(MAIN).log 2>/dev/null | grep -qv '^0$$' \
	    && echo "    WARNING: undefined references" || true; \
	  grep -q 'Overfull \\\\hbox' $(MAIN).log 2>/dev/null \
	    && echo "    note: overfull hbox, check the margins" || true; \
	fi

clean:
	@rm -rf $(JUNK) build/
	@rm -f $(MAIN)-diff.tex $(MAIN)-diff.pdf

distclean: clean
	@rm -f $(MAIN).pdf
	@echo "removed $(MAIN).pdf"

# --- figures ---------------------------------------------------------------

figures:
	@for f in figures/src/*.drawio; do \
	  [ -e "$$f" ] || continue; \
	  bash $(SKILL)/scripts/drawio-export.sh "$$f" figures/; \
	done

# Verify every figure's text lands at caption_pt from .paper-meta.yml.
check:
	@cap=$$(grep '^caption_pt:' $(META) | sed 's/[^0-9.]//g'); \
	 [ -n "$$cap" ] || { echo "no caption_pt in $(META)"; exit 1; }; \
	 echo "target $$cap pt"; \
	 for f in figures/*.pdf; do \
	   [ -e "$$f" ] || continue; \
	   echo "-- $$f"; \
	   python3 $(SKILL)/scripts/figfont.py check "$$f" --target $$cap || true; \
	 done

lint:
	@python3 $(SKILL)/scripts/lint.py --paper $(MAIN).tex sections/*.tex

# --- review ----------------------------------------------------------------

REV ?= HEAD~1
diff:
	@bash $(SKILL)/scripts/paper-diff.sh $(REV) HEAD

watch:
	@command -v fswatch >/dev/null || { echo "brew install fswatch"; exit 1; }
	@echo "watching. ctrl-c to stop."
	@fswatch -o $(MAIN).tex sections tables results.tex refs.bib | \
	  while read -r _; do $(MAKE) --no-print-directory pdf; done

help:
	@sed -n '3,12p' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'
