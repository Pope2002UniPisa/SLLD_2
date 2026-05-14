%% MasterSTEP.cls - Version 1 - Mars 2012 - AF et GM
%% MasterSTEP.cls - Version 1.2 - pxfont + hyperref
%% MasterGGG.cls anefine MasterGGG.cls - Version 1.5 - new logos (un par mention) 
%% Un Master.cls pour les 2 masters - Version 1.6
%% Un ipgpmaster.cls pour les 2 masters - Mai 2020 - Avril 2025

\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{ipgpmaster}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,twoside]{article}


\def\mymaster{IPGPUPC}


%---- Spacing around titles
\renewcommand\section{\@startsection {section}{1}{\z@}%
%     {-3.5ex \@plus -1ex \@minus -.2ex}%  before skip
%     {2.3ex \@plus.2ex}% after skip
     {-1.2ex \@plus -1ex \@minus -.2ex}%  before skip
     {0.6ex \@plus.2ex}% after skip
     {\normalfont\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
%     {-3.25ex\@plus -1ex \@minus -.2ex}% 
%     {1.5ex \@plus .2ex}%
     {-2.25ex\@plus -1ex \@minus -.2ex}% 
     {1.0ex \@plus .2ex}%
     {\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
%     {-3.25ex\@plus -1ex \@minus -.2ex}%
%     {1.5ex \@plus .2ex}%
     {-2.0ex\@plus -1ex \@minus -.2ex}%
     {0.75ex \@plus .2ex}%
     {\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
     {1.0ex \@plus .5ex \@minus .2ex}%
     {-1em}%
     {\normalfont\normalsize\bfseries}}

%---- Table of Contents and sectioning  depth
\setcounter{secnumdepth}{3} 
\setcounter{tocdepth}{2}


\RequirePackage{graphicx} 
\RequirePackage[x11names]{xcolor}

\RequirePackage[pdfstartview=FitH,%
  bookmarks=true,%
  bookmarksopen=true,%
  breaklinks=true,%
  colorlinks=true,%
  linkcolor=blue,anchorcolor=blue,%
  citecolor=blue,filecolor=blue,%
  menucolor=blue,%
  urlcolor=blue]{hyperref}


\RequirePackage{booktabs}  % Prettier tables
%\RequirePackage{palatino} % Pretty font
\RequirePackage[scaled=.90]{helvet}
%\RequirePackage{pxfonts,txfonts} % clash with amsmath
\RequirePackage{newtxtext,newtxmath}
\RequirePackage{lineno} % Possibility to add line numbers
\RequirePackage[a4paper,margin=2.5cm]{geometry}
%\RequirePackage[latin1]{inputenc}
\parskip 8pt plus1pt minus1pt
\parindent 0pt

%---- Title page
\def\author{}
\def\title{}
\def\shorttitle{}
\def\spe{}
\def\unit{}
\def\team{}
\def\supervisor{}
\def\keywords{}

\newcounter{firstyear}
\newcounter{lastyear}[firstyear]
\newcounter{thismonth}

\newcommand{\checkyears}{
\setcounter{firstyear}{\the\year}
\setcounter{thismonth}{\the\month}
%\setcounter{thismonth}{7}
\ifnum\the\value{thismonth}<9
   \addtocounter{firstyear}{-1}
\fi
\setcounter{lastyear}{\value{firstyear}}
\stepcounter{lastyear}
}


\newcommand{\yearlabel}{Année Universitaire }
\newcommand{\unitlabel}{Unité de Recherche : }
\newcommand{\teamslabel}{Course: }
\newcommand{\suplabel}{Encadrant(s) : }
\newcommand{\abstractlabel}{Résumé}
\newcommand{\keyslabel}{Mots clés : }
\newcommand{\mydate}{}


\newcommand{\Entete}{

 \IfLanguagePatterns{english}{
    \renewcommand{\yearlabel}{Academic Year }
    \renewcommand{\unitlabel}{University: }
    \providecommand{\classlabel}{Class: }
    \providecommand{\modulelabel}{Module: }
    \renewcommand{\suplabel}{Supervisor(s): }
    \renewcommand{\abstractlabel}{Abstract}
    \renewcommand{\keyslabel}{Keywords: }
    }{}


\vspace*{-2cm}

\begin{minipage}{0.6\linewidth}
\yearlabel \thefirstyear\, -- \thelastyear\\
\unitlabel  \unit\\
\teamslabel \team\\
\suplabel  \supervisor\\
%\today\\
\end{minipage}
\hfill
\begin{minipage}{0.30\linewidth}
\includegraphics[width=\linewidth]{logosantanna.png}\\[2mm]
\hspace*{\fill}{\sc \spe}\hspace*{1.5mm}\\[1cm]
\end{minipage}

{\parskip 10pt
\begin{center}
\hrule
\vspace*{1mm}
{\LARGE\bfseries \title}\\[5mm]
\hrule
\vspace*{4mm}
{\large\mdseries \author}\\[3mm]
\mydate
\end{center}
}}

%---- Headers and footers
\RequirePackage{fancyhdr}
\addtolength{\headheight}{1.6pt} % si 11pt
\lhead[\fancyplain{}{\bfseries\thepage}]{\fancyplain{}{\author}}
\rhead[\fancyplain{}{\nouppercase{\bfseries \shorttitle}}]%
{\fancyplain{}{\bfseries\thepage}}
%\cfoot{\small Année 2011 -- 2012}
\fancyfoot{}
\fancyfoot[LE,RO]{\itshape \yearlabel \thefirstyear\, -- \thelastyear}
%\renewcommand{\footrulewidth}{0.4pt}
\pagestyle{fancyplain}

%---- Abstract and Keywords
\renewenvironment{abstract}
{
\thispagestyle{empty}
\begin{center}
{\bfseries \abstractlabel}
\end{center}
\small
}
{
\par
{\bfseries \keyslabel}  \keywords
\par}

%---- French
\RequirePackage[english,french]{babel} % Last language, language by default
\RequirePackage{iflang}
\def\frenchcontentsname{Sommaire}

\usepackage[hypcap=true]{caption}

%---- Captions
\long\def\@caption#1[#2]#3{\par\addcontentsline{\csname
  ext@#1\endcsname}{#1}{\protect\numberline{\csname
  the#1\endcsname}{\ignorespaces #2}}\begingroup
    \@parboxrestore
    \normalsize
    \expandafter\let\expandafter\@tempa\csname @make#1caption\endcsname
    \ifx\@tempa\relax\let\@tempa\@makecaption\fi
    \@tempa{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
  \endgroup}
%
% Here is the content of a .sty file containing nebinitions using the above
% hook.
%
% The following is the same as the \@makecaption in book.sty:
\long\def\@makefigurecaption#1#2{%
 \vskip 10pt 
 \setbox\@tempboxa\hbox{#1 -- {\small #2}}%
 \ifdim \wd\@tempboxa >\hsize #1 -- {\small #2}\par \else
  \hbox to\hsize{\hfil\box\@tempboxa\hfil} %
  \fi
 }
%
% The following is slightly different from the \@makecaption in book.sty:
\long\def\@maketablecaption#1#2{%
 \setbox\@tempboxa\hbox{#1 -- {\small #2}}%
 \ifdim \wd\@tempboxa >\hsize #1 -- {\small #2}\par \else
  \hbox to\hsize{\hfil\box\@tempboxa\hfil} %
  \fi
 \vskip 10pt 
 }
 
%---- Codes
\usepackage{listings}
\lstdefinestyle{mystyle}{
breakatwhitespace=false,
    commentstyle=\color{gray},
    keywordstyle=\color{Green4},
    numberstyle=\footnotesize\color{gray},
    stringstyle=\color{purple},
    identifierstyle=\color{SteelBlue4},
    basicstyle=\ttfamily,
    numbers=left,
    keepspaces=false,    
    numbers=left,    
    showspaces=false,    
    showstringspaces=true,
    }   
\lstset{style=mystyle}

\usepackage{emptypage} % For the last page

%---- Bibliographies
%\bibliographystyle{plainnat}
%\RequirePackage{hyperref,doi}
%\usepackage[citestyle=alphabetic,bibstyle=authortitle,sorting=ydnt,natbib=true]{biblatex}
\RequirePackage{etoolbox}
\RequirePackage{hyperref}
\RequirePackage[autostyle=false]{csquotes}
%\RequirePackage[backend=biber,natbib,
\RequirePackage[backend=biber,
style=authoryear,
citestyle=authoryear-comp,
giveninits=true,       % initiales dans la biblio
maxcitenames=2,        % jusqu'à 2 noms dans la citation
mincitenames=1,        % troncature à 1 nom si plus
maxbibnames=99,        % tous les noms dans la biblio
minbibnames=1,
url=false,
uniquename=false,
uniquelist=false,
hyperref]{biblatex}

\DeclareNameAlias{author}{family-given} % initiales dans la biblio

\DeclareNameFormat{labelname}{%
  \nameparts{#1}%
  \printtext{\namepartfamily}%
  % "et" uniquement si exactement 2 auteurs
  \ifthenelse{\value{liststop}=2 \AND \value{listcount}=1}{\addspace et\space}{}%
  \usebibmacro{name:andothers}% gère "et al."
}


% Pas de smallcaps dans les noms d'auteurs
% \DefineBibliographyExtras{french}{\restorecommand\mkbibnamefamily}

%\DeclareCiteCommand{\cite} % -> Moguilny [2025]
%  {\usebibmacro{prenote}}
%  {\bibhyperref{%
%    \usebibmacro{citeindex}%
%    \printnames{labelname}%
%    \addspace\mkbibbrackets{\printfield{year}}%
%  }}
%  {\multicitedelim}
%  {\usebibmacro{postnote}}

%\DeclareCiteCommand{\parencite}[\mkbibbrackets] % ->  [Roberts et Aurnou (2012)]
%  {\usebibmacro{prenote}}
%  {\bibhyperref{%
%    \usebibmacro{citeindex}%
%    \printnames{labelname}%
%    \addspace\mkbibbrackets{\printfield{year}}%
%  }}  
%  {\multicitedelim}
%  {\usebibmacro{postnote}}


\renewbibmacro*{prenote}{%
  \iffieldundef{prenote}
    {}
    {\printfield{prenote}\addspace}}

\DeclareCiteCommand{\parencite} % ->  (Roberts et Aurnou, 2012)
  {}
  {\mkbibparens{%
     \iffieldundef{prenote}
       {}
       {\printfield{prenote}\addspace}%
     \bibhyperref{%
       \usebibmacro{citeindex}%
       \printnames{labelname}%
       \setunit{\nameyeardelim}%
       \printfield{year}}%
     \iffieldundef{postnote}
       {}
       {\addcomma\space\printfield{postnote}}}}
  {\addsemicolon\space}
  {}


\DeclareCiteCommand{\cite} % ->  (Roberts et Aurnou, 2012)
  {\usebibmacro{prenote}}
  {\mkbibparens{%
     \bibhyperref{%
       \usebibmacro{citeindex}%
       \printnames{labelname}%
       \setunit{\nameyeardelim}%
       \printfield{year}}}}
  {\multicitedelim}
  {\setunit{\addcomma\space}\usebibmacro{postnote}}



\DeclareCiteCommand{\textcite}
  {\usebibmacro{prenote}}
  {\bibhyperref{\usebibmacro{citeindex}%
               \printnames{labelname}% 
               \space (\printfield{year})}}% 
  {\multicitedelim}
  {\usebibmacro{postnote}}

\DeclareCiteCommand{\citeauthor}
  {\boolfalse{citetracker}%
   \boolfalse{pagetracker}%
   \usebibmacro{prenote}}
  {\bibhyperref{\usebibmacro{citeindex}%
   \printnames{labelname}}} %
  {\multicitedelim}
  {\usebibmacro{postnote}}

\DeclareCiteCommand{\citeyear}
  {\boolfalse{citetracker}%      
   \boolfalse{pagetracker}%      
   \usebibmacro{prenote}}        
  {\bibhyperref{\printfield{year}}} 
  {\multicitedelim}              
  {\usebibmacro{postnote}}     


\renewbibmacro*{doi+eprint+url}{%
  \printfield{doi}%
  \newunit\newblock
  \printfield{eprint}%
  \newunit\newblock
  \iffieldundef{doi}{%
    \printfield{url}%
    \printfield{urldate}%
  }{}%
}


\setlength{\bibitemsep}{0.9\baselineskip}
\DeclareDelimFormat{nameyeardelim}{\addcomma\space}
