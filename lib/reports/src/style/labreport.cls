%%============================================================================
%% labreport.cls --- one research report, five pages
%%
%% One report is one file under lib/reports/src/, compiled from the repository root:
%%     report-build.sh lib/reports/src/<YYYY-MM-DD>-<author>-<headline>.tex
%%
%% Engine: pdflatex. Font: lmodern. Nothing here needs a font or a package the
%% machine does not already ship, so the report compiles on the first try, on a
%% laptop or in Overleaf.
%%
%% The contract this file enforces:
%%   the headline is assembled, never typed
%%   section 0, when the repository asks for one, is read from
%%     lib/reports/reflections/<jobname>.tex and nowhere else
%%   the section names are commands, so they cannot drift
%%   the report is five pages
%%============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{labreport}[2026-09-09 v1.0 research report]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,a4paper,oneside]{article}

%%--- packages ---------------------------------------------------------------
%% Required, and every one of these ships with a basic TeX install (TeX Live
%% scheme-basic, which is what MacTeX BasicTeX and most Linux texlive-latex-base
%% packages give you). A teammate should never have to run tlmgr to read or
%% write a report.
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{amsmath}
\RequirePackage[a4paper,margin=2.3cm,headheight=15pt,headsep=9pt,footskip=1.1cm]{geometry}
\RequirePackage{graphicx}
\RequirePackage{array}
\RequirePackage[hidelinks,breaklinks]{hyperref}

%% Colour. xcolor when it is there, the graphics bundle's own color package
%% when it is not. Every colour below is defined in rgb, which both understand,
%% so the palette does not depend on which one loaded.
\newif\iflr@xcolor
\IfFileExists{xcolor.sty}{\RequirePackage{xcolor}\lr@xcolortrue}
                         {\RequirePackage{color}}

%% Optional. Each one makes the page better and none is needed to build. On a
%% minimal TeX install the report compiles and comes out slightly plainer.
\newif\iflr@listings
\newif\iflr@pgfplots
\IfFileExists{lmodern.sty}  {\RequirePackage{lmodern}}{}
\IfFileExists{microtype.sty}{\RequirePackage{microtype}}{}
\IfFileExists{listings.sty} {\RequirePackage{listings}\lr@listingstrue}{}
\iflr@xcolor
  \IfFileExists{pgfplots.sty}{\RequirePackage{pgfplots}\lr@pgfplotstrue
                              \pgfplotsset{compat=1.18}}{}
\fi

%% booktabs draws three rules and lives outside a basic install, so the format
%% carries its own at the same weights rather than depending on it.
\IfFileExists{booktabs.sty}{\RequirePackage{booktabs}}{%
  \newcommand{\toprule}   {\noalign{\vskip1pt\hrule height0.8pt\vskip3pt}}
  \newcommand{\midrule}   {\noalign{\vskip3pt\hrule height0.4pt\vskip3pt}}
  \newcommand{\bottomrule}{\noalign{\vskip3pt\hrule height0.8pt\vskip1pt}}}

\setlength{\parskip}{0.25em}
\setlength{\parindent}{0pt}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.15}

%%--- palette ----------------------------------------------------------------
%% lrblue through lrgray are also the categorical series in labreport.mplstyle
%% and, when pgfplots is installed, in the lraxis style. Change a colour here
%% and change it there. rgb rather than HTML, so the color package can read it.
\definecolor{lrnavy}  {rgb}{0.110,0.239,0.353}
\definecolor{lrblue}  {rgb}{0.180,0.424,0.643}
\definecolor{lrteal}  {rgb}{0.078,0.490,0.482}
\definecolor{lramber} {rgb}{0.698,0.349,0.000}
\definecolor{lrred}   {rgb}{0.753,0.224,0.169}
\definecolor{lrviolet}{rgb}{0.420,0.298,0.604}
\definecolor{lrgray}  {rgb}{0.431,0.482,0.545}
\definecolor{lrrule}  {rgb}{0.725,0.769,0.816}
\definecolor{lrcode}  {rgb}{0.965,0.973,0.980}

%%--- metadata ---------------------------------------------------------------
%% A report's identity is its file name, <date>-<slug>.tex, where the slug is
%% <author>-<headline> and the author is a one-word handle, and \lrdate and
%% \lrslug are the same two strings declared inside the file. report-check.py
%% recomputes the name from them and fails when the two disagree, so a report
%% cannot be renamed out of step with what it says it is.
%%
%% \lrtitle is the sentence a person reads. \lrissue names the issue the report
%% answers, by its file name in lib/issues/ or by number, and is optional only for work that no issue
%% asked for.
\newcommand{\lr@date}  {\ClassError{labreport}{No \string\lrdate}{Declare it in the preamble, as YYYY-MM-DD.}}
\newcommand{\lr@slug}  {\ClassError{labreport}{No \string\lrslug}{Declare it in the preamble, as <author>-<headline>. It is the file name after the date.}}
\newcommand{\lr@title} {\ClassError{labreport}{No \string\lrtitle}{Declare it in the preamble.}}
\newcommand{\lr@author}{\ClassError{labreport}{No \string\lrauthor}{Declare it in the preamble.}}
\newcommand{\lr@status}{draft}
\newcommand{\lr@issue}{}
\newcommand{\lr@period}{}

\newcommand{\lrdate}  [1]{\renewcommand{\lr@date}{#1}}
\newcommand{\lrslug}  [1]{\renewcommand{\lr@slug}{#1}}
\newcommand{\lrtitle} [1]{\renewcommand{\lr@title}{#1}}
\newcommand{\lrauthor}[1]{\renewcommand{\lr@author}{#1}}
\newcommand{\lrstatus}[1]{\renewcommand{\lr@status}{#1}}
\newcommand{\lrissue} [1]{\renewcommand{\lr@issue}{#1}}
\newcommand{\lrperiod}[1]{\renewcommand{\lr@period}{#1}}

%% \lrclaims{slug-one,slug-two} names the claims this report is evidence for.
%% \lrresults{lab/results/<topic>} names the directory its numbers were read
%% from. Each prints one line under the metadata block, and report-check.py
%% reads both back. A results path that does not exist fails the check. A claim
%% slug is recorded and not checked here, because the claim ledger is what
%% checks a claim.
\newcommand{\lr@claims}{}
\newcommand{\lr@results}{}
\newcommand{\lrclaims} [1]{\renewcommand{\lr@claims}{#1}}
\newcommand{\lrresults}[1]{\renewcommand{\lr@results}{#1}}

%% Section 0 is on unless the report turns it off with \lrnoreflection, which
%% report-new.sh writes from lib/reports/src/config.yml.
\newif\iflr@reflection \lr@reflectiontrue
\newcommand{\lrnoreflection}{\lr@reflectionfalse}

\newcommand{\lrheadline}{\lr@title{} - \lr@author}
\newcommand{\lr@stem}{\lr@date-\lr@slug}

%%--- a file path, safe with underscores --------------------------------------
\newcommand{\lrpath}[1]{\texttt{\small\detokenize{#1}}}

%%--- page furniture ---------------------------------------------------------
%% Defined here rather than with fancyhdr, which is one more package to install
%% for two lines of output. The headline is carried by the running header, on
%% every page, because the title block that used to repeat it on page one was
%% mostly white space.
\newcommand{\lr@rulebox}[1]{\hbox to\textwidth{\color{lrrule}\rule{\textwidth}{#1}}}
\def\ps@labreport{%
  \def\@oddhead{\vbox{%
    \hbox to\textwidth{\footnotesize\lrheadline\hfill\footnotesize\lr@date}%
    \vskip2pt\lr@rulebox{0.4pt}}}%
  \def\@oddfoot{\hbox to\textwidth{%
    \footnotesize\color{lrgray}\lr@stem\quad\lr@status\hfill
    \footnotesize\color{lrgray}\thepage\ (limit \lr@pagelimit)}}%
  \let\@evenhead\@oddhead
  \let\@evenfoot\@oddfoot}
\pagestyle{labreport}

%%--- headings ---------------------------------------------------------------
%% \@startsection rather than titlesec, and \@makecaption rather than caption.
%% Both are core LaTeX and both do everything this format asks of them.
\renewcommand\section{\@startsection{section}{1}{\z@}%
  {-0.85em \@plus -0.2em \@minus -0.1em}{0.25em \@plus 0.1em}%
  {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-0.6em \@plus -0.2em}{0.2em}{\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-0.5em \@plus -0.2em}{0.2em}{\normalfont\normalsize\itshape}}

%% Caption above the object, label in bold, a period after it.
\long\def\@makecaption#1#2{%
  \vskip4\p@
  \sbox\@tempboxa{\small\textbf{#1.} #2}%
  \ifdim\wd\@tempboxa>\hsize
    {\small\textbf{#1.} #2\par}%
  \else
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip4\p@}

\hypersetup{colorlinks=true,linkcolor=lrnavy,citecolor=lrnavy,urlcolor=lrblue}

%%--- floats -----------------------------------------------------------------
%% A five page report cannot afford a float drifting two pages from the
%% sentence that refers to it, so the page may be nearly all float.
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.6}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}

%%--- the five page limit ----------------------------------------------------
%% report-build.sh is the authority and reads the page count from pdfinfo.
%% This is the courtesy copy, for a reader compiling in Overleaf with no build
%% script.
\newcommand{\lr@pagelimit}{5}
\newcommand{\lr@setpages}[1]{\gdef\lr@pages{#1}}
\AtEndDocument{\protected@write\@auxout{}{\string\lr@setpages{\thepage}}}
\AtBeginDocument{%
  \@ifundefined{lr@pages}{}{%
    \ifnum\lr@pages>\lr@pagelimit\relax
      \ClassWarningNoLine{labreport}{^^JThis report is \lr@pages\space pages and the
        limit is \lr@pagelimit. Cut it before it goes into the knowledge base.^^J}%
    \fi}}

%%--- the opening ------------------------------------------------------------
%% There is no title block. The headline is in the running header on every page
%% and in the PDF metadata, and repeating it on page one bought nothing but
%% white space. Status sits in the footer. Date, status and period stay in the
%% source, which is where report-index.py reads them from.
\newcommand{\lrmaketitle}{%
  \hypersetup{pdftitle={\lrheadline},pdfauthor={\lr@author},
              pdfsubject={Research report},
              pdfkeywords={\lr@stem}}%
  \thispagestyle{labreport}%
  %% Section 0 opens the report when there is one, so nothing floats above it.
  \suppressfloats[t]%
  \lr@pointers
  \iflr@reflection
    \setcounter{section}{-1}%
    \lr@reflection
  \else
    \setcounter{section}{0}%
  \fi
}

%%--- what this report answers and where its numbers came from ----------------
%% Two lines, small and grey, above whatever section opens the report. They are
%% the seam between a report and the rest of the repository: the claims it is
%% evidence for, and the results directory it read. Nothing prints when neither
%% is declared.
\newcommand{\lr@pointers}{%
  \ifx\lr@claims\@empty\else
    {\small\color{lrgray}Claims: \lr@claims}\par
  \fi
  \ifx\lr@results\@empty\else
    {\small\color{lrgray}Results:
     \expandafter\lrpath\expandafter{\lr@results}}\par
  \fi
  \ifx\lr@claims\@empty\ifx\lr@results\@empty\else\vspace{0.3em}\fi
  \else\vspace{0.3em}\fi
}

%%--- section 0, and the one file it may come from ----------------------------
%% Read from lib/reports/reflections/<jobname>.tex. The pairing is structural:
%% a report named 2026-09-16-x.tex reads
%% lib/reports/reflections/2026-09-16-x.tex and cannot be pointed anywhere
%% else. Nothing under reflections/ is written by a language model.
%% report-attest.sh signs it and report-build.sh checks the signature.
%%
%% \lrnoreflection turns the section off for a repository whose reports carry
%% no section 0. report-new.sh writes it from lib/reports/src/config.yml, so the
%% two cannot disagree, and report-check.py fails when they do.
\newcommand{\lr@reflection}{%
  \section{REFLECTION}\label{lrpage:\thesection}%
  \IfFileExists{lib/reports/reflections/\jobname.tex}%
    {\input{lib/reports/reflections/\jobname.tex}}%
    {\ClassError{labreport}{No lib/reports/reflections/\jobname.tex}%
      {Section 0 is written by the author. Run report-new.sh, which creates
       the file, then write it and run report-attest.sh. A repository that
       wants no section 0 sets reflection: false in lib/reports/src/config.yml,
       and report-new.sh then writes \string\lrnoreflection into the report.}}%
}

%%--- the required sections and the optional blocks ---------------------------
%% Every section is a command, so a name cannot drift between two reports and
%% check-report.py knows what it is reading. The page label on each one is what
%% build.sh reads to say where the five pages went.
\newcommand{\lr@sec}[1]{\section{#1}\label{lrpage:\thesection}}
\newcommand{\lrscope}    {\lr@sec{Scope}}
\newcommand{\lrresult}   {\lr@sec{Result}}
\newcommand{\lrproduced} {\lr@sec{How it was produced}}
\newcommand{\lrchanges}  {\lr@sec{What this changes}}
\newcommand{\lrmethod}   {\lr@sec{Method}}
\newcommand{\lrfailures} {\lr@sec{Failure log}}
\newcommand{\lrcost}     {\lr@sec{Cost}}
\newcommand{\lrnext}     {\lr@sec{Next}}

%%--- the boundary line ------------------------------------------------------
%% Every Result carries one. A result with no stated boundary is a claim with
%% no scope. Plain text, no emphasis, because the sentence carries itself.
\newcommand{\lrboundary}[1]{%
  \par\vspace{0.4em}%
  {\color{lrrule}\rule{0.18\linewidth}{0.4pt}}\par\vspace{0.1em}%
  {\small Does not show: #1}\par\vspace{0.35em}}

%%--- facts about the run ----------------------------------------------------
%% "How it was produced" is a table, not prose. Anything that is a fact about
%% the run lives here, which is what keeps the prose inside its budget.
%% Plain tabular, not tabularx: tabularx scans the input stream for its own
%% \end and cannot be split across a \newenvironment pair.
\newenvironment{lrfacts}%
  {\par\vspace{0.2em}\small
   \setlength{\parskip}{0pt}%
   \tabular{@{}p{0.20\linewidth}%
              >{\raggedright\arraybackslash}p{\dimexpr0.80\linewidth-2\tabcolsep\relax}@{}}\toprule}%
  {\bottomrule\endtabular\par\vspace{0.3em}}
%% \textcolor, not \color: a bare \color at the head of a p-column cell
%% opens an empty first line and the label then sits one line low.
\newcommand{\lrfact}[2]{\textbf{#1} & #2 \\}

%%--- one number, one place --------------------------------------------------
%% \lrdefine{Speedup}{2.11$\times$}{data/<stem>-throughput.csv} defines the
%% number and records where it came from. Prose writes \lrv{Speedup} and never
%% the literal, so the number cannot say two different things in one report.
%% \lrprovenance prints what was defined, with sources.
%% Rows are baked in as \lrdefine runs, not looped over at print time: an
%% \@for loop cannot build table rows, because every cell is a group and the
%% loop variable is gone before the cell is typeset.
\newcommand{\lr@rows}{}
\newcommand{\lr@provrow}[1]{%
  #1 & \csname lrv@#1\endcsname &
  \texttt{\footnotesize\csname lrs@#1\endcsname} \\}
\newcommand{\lrdefine}[3]{%
  \expandafter\gdef\csname lrv@#1\endcsname{#2}%
  \expandafter\xdef\csname lrs@#1\endcsname{\detokenize{#3}}%
  \g@addto@macro\lr@rows{\lr@provrow{#1}}}
\newcommand{\lrv}[1]{%
  \@ifundefined{lrv@#1}%
    {\ClassError{labreport}{Result '#1' is used but never defined}%
      {Define it with \string\lrdefine{#1}{value}{source file}.}}%
    {\csname lrv@#1\endcsname}}
\newcommand{\lrprovenance}{%
  \ifx\lr@rows\@empty\else
    \par\vspace{0.9em}%
    {\small\bfseries Numbers in this report}\par\vspace{0.15em}%
    \small
    \setlength{\parskip}{0pt}%
    \begin{tabular}{@{}l l >{\raggedright\arraybackslash}p{0.60\linewidth}@{}}\toprule
      \textbf{Number} & \textbf{Value} & \textbf{Source} \\ \midrule
      \lr@rows
      \bottomrule
    \end{tabular}\par\vspace{0.3em}
  \fi}

%%--- a figure, sized to the text --------------------------------------------
\newcommand{\lrfigure}[3]{%
  \begin{figure}[htbp]\centering
    \includegraphics[width=\linewidth]{lib/reports/src/figures/#1}
    \caption{#2}\label{fig:#3}
  \end{figure}}
\newcommand{\lrfigurew}[4]{%
  \begin{figure}[htbp]\centering
    \includegraphics[width=#4\linewidth]{lib/reports/src/figures/#1}
    \caption{#2}\label{fig:#3}
  \end{figure}}

%%--- a mean with its dispersion ---------------------------------------------
%% Never a mean on its own. \lrpm{12.4}{0.3} sets 12.4 +/- 0.3 at the right size.
\newcommand{\lrpm}[2]{$#1\,{\pm}\,#2$}

%%--- pgfplots, matched to labreport.mplstyle ---------------------------------
%% Only when pgfplots is installed. It is not in a basic TeX install and it is
%% the one optional thing that changes what an author can write, so a report
%% that uses lraxis will not build for a teammate without it. matplotlib
%% through plotstyle.py is the path that always works. See references/plots.md.
\iflr@pgfplots
\pgfplotsset{
  lraxis/.style={
    width=\linewidth, height=0.62\linewidth,
    font=\footnotesize,
    axis line style={draw=black!70,line width=0.7pt},
    tick style={draw=black!70,line width=0.7pt},
    grid=both, grid style={draw=lrrule,dotted,line width=0.5pt},
    label style={font=\footnotesize}, tick label style={font=\scriptsize},
    title style={font=\footnotesize,at={(0,1)},anchor=south west},
    legend style={font=\scriptsize,draw=lrrule,fill=white,
                  legend cell align=left,inner sep=3pt},
    every axis plot/.append style={line width=1.2pt,mark size=1.7pt},
    cycle list={
      {lrblue,mark=*},{lrteal,mark=square*},{lramber,mark=triangle*},
      {lrred,mark=diamond*},{lrviolet,mark=pentagon*},{lrgray,mark=x}},
  }}
\fi

%%--- code, when a report has to show a command -------------------------------
\iflr@listings
\lstdefinestyle{lrcode}{
  basicstyle=\ttfamily\footnotesize, backgroundcolor=\color{lrcode},
  frame=single, rulecolor=\color{lrrule}, framesep=4pt,
  breaklines=true, showstringspaces=false,
  keywordstyle=\color{lrnavy}, commentstyle=\color{lrgray},
  stringstyle=\color{lrteal}, numbers=none, xleftmargin=0pt}
\lstset{style=lrcode}
\fi

%%--- references -------------------------------------------------------------
%% One shared bibliography at lib/reports/src/style/labreport.bib. \lrreferences prints it as an
%% unnumbered section, which keeps it out of the numbered spine.
\newcommand{\lrreferences}{%
  \small
  \bibliographystyle{plain}
  \renewcommand{\refname}{References}
  \bibliography{lib/reports/src/style/labreport}}

\endinput
