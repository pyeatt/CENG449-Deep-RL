\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{exam}[1999/2/20, Version 0.1Beta, Larry D. Pyeatt]


\RequirePackage{ifthen}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions\relax

\LoadClass{article}

\newboolean{@answers}
\setboolean{@answers}{false}


\newcommand{\answers}
{\setboolean{@answers}{true}}

\newcommand{\answer}[2]{
\ifthenelse{\boolean{@answers}}
{\par\begin{minipage}[t][#1]{\textwidth}
\ \par\noindent{\bf #2}
\end{minipage}\hfill}
{\vspace{#1}}\pagebreak[3]}

\newlength{\qwidth}
\newcounter{qc}
\newcounter{sqc}[qc]

%% \newcommand{\question}[2]{\par\noindent\stepcounter{qc}
%% \arabic{qc}.\hfill
%% \setlength{\qwidth}{\textwidth}
%% \addtolength{\qwidth}{-0.4in}
%% \begin{minipage}[t]{\qwidth} (#1 pts) #2\end{minipage}\nopagebreak}

\newenvironment{question}[1]{\par\noindent\stepcounter{qc}
\vspace{2 \baselineskip}
\arabic{qc}.\hfill
\setlength{\qwidth}{\textwidth}
\addtolength{\qwidth}{-0.4in}
\begin{minipage}[t]{\qwidth}(#1 pts) }
{\end{minipage}\nopagebreak}

\newenvironment{subquestion}{\par\noindent
\stepcounter{sqc}
\alph{sqc})\hfill
\setlength{\qwidth}{\textwidth}
\addtolength{\qwidth}{-0.4in}
\begin{minipage}[t]{\qwidth}}
{\end{minipage}\nopagebreak}

%\newcommand{\subquestion}[1]{\par\noindent\hspace{0.4in}\stepcounter{sqc}
%\alph{sqc})\hfill
%\setlength{\qwidth}{\textwidth}
%\addtolength{\qwidth}{-1in}
%\hspace{0.1in}\begin{minipage}[t]{\qwidth} #1\end{minipage}\nopagebreak}

\def\@maketitle{%
  \newpage
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em
\noindent Name: \rule{2.5in}{0.5pt} ID\#: \rule{1.5in}{0.5pt} 
\hfill Score: \rule{0.5in}{0.5pt}
\par
\vspace{0.25in}
}

\setlength{\parindent}{0in}

\textwidth 6.5in
\textheight 9in
\evensidemargin 0in
\oddsidemargin 0in
\headheight 0in
\headsep 0in

