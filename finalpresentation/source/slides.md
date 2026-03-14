---
title: Agentic AI in FinTech
author: Eugenio Rivera Ramos \& Oguz S.
---

<!-- center -->
{\Huge\bfseries\color{accent} Agentic AI in FinTech\par}
\vspace{0.4em}
{\large\color{muted} Single-Agent vs Multi-Agent for Financial Reasoning\par}
<!-- /center -->

```tikz
\begin{tikzpicture}[remember picture, overlay]
    \only<3->{
        \node[anchor=south, yshift=10mm, text=text] at (current page.south) {\Large\itshape Eugenio Rivera Ramos \quad\&\quad Oguz S.};
    }
\end{tikzpicture}
```

---

## Problem Overview

<!-- center -->
**Can LLMs answer live financial questions reliably?**
<!-- /center -->

- Financial reasoning requires real-time data: prices, P/E ratios, sector lookups, news sentiment
- Training knowledge is stale --- hallucination risk is high without grounding
- 15 benchmark questions (easy / medium / hard) evaluated across Baseline, Single-Agent, and Multi-Agent
- Two independent implementations compared on gpt-4o-mini and gpt-4o

\begin{block}{Core Question}
When does adding agent complexity actually improve over a bare LLM call?
\end{block}

---

## System Architecture --- Overview

```tex
\begin{tikzpicture}[
    scale=0.78, transform shape,
    box/.style={draw=accent, fill=bg, text=text, rounded corners=3pt,
                minimum width=22mm, minimum height=9mm, align=center, font=\small},
    sbox/.style={draw=accent2, fill=bg, text=text, rounded corners=3pt,
                 minimum width=19mm, minimum height=9mm, align=center, font=\small},
    arr/.style={-Stealth, color=muted, semithick},
    lbl/.style={font=\small\bfseries, color=muted, anchor=east},
]
% ── Row 1: Baseline ─────────────────────────────────────────
\node[lbl] (lb) at (0,0)        {Baseline};
\node[box, right=4mm of lb]     (b_q)   {User Q};
\node[box, right=7mm of b_q]    (b_llm) {LLM\\\scriptsize no tools};
\node[box, right=7mm of b_llm]  (b_a)   {Answer};
\draw[arr] (b_q)--(b_llm);
\draw[arr] (b_llm)--(b_a);

% ── Row 2: Single Agent ─────────────────────────────────────
\node[lbl] (ls) at (0,-18mm)    {Single};
\node[box, right=4mm of ls]     (s_q)   {User Q};
\node[box, right=7mm of s_q]    (s_ag)  {Agent\\\scriptsize 7 tools};
\node[box, right=7mm of s_ag]   (s_a)   {Answer};
\draw[arr] (s_q)--(s_ag);
\draw[arr] (s_ag)--(s_a);

% ── Row 3: Multi-Agent ──────────────────────────────────────
\node[lbl] (lm) at (0,-36mm)    {Multi};
\node[box, right=4mm of lm]     (m_q)   {User Q};
\node[box, right=7mm of m_q]    (rtr)   {Router\\\scriptsize adaptive};
\node[sbox, above right=3mm and 7mm of rtr]  (mkt)  {Market};
\node[sbox, right=7mm of rtr]                (fund) {Fund.};
\node[sbox, below right=3mm and 7mm of rtr]  (news) {News};
\node[box, right=7mm of fund]   (verif) {Verifier};
\node[box, right=7mm of verif]  (agg)   {Aggregator};
\node[box, right=7mm of agg]    (m_a)   {Answer};
\draw[arr] (m_q)--(rtr);
\draw[arr] (rtr)--(mkt);
\draw[arr] (rtr)--(fund);
\draw[arr] (rtr)--(news);
\draw[arr] (mkt) -| (verif);
\draw[arr] (fund)-- (verif);
\draw[arr] (news)-| (verif);
\draw[arr] (verif)--(agg);
\draw[arr] (agg)  --(m_a);
\end{tikzpicture}
```

---

## Key Components --- 7 Tools

```tex
\begin{itemize}
\item \texttt{get\_price\_performance} — yfinance historical prices
\item \texttt{get\_top\_gainers\_losers} — top movers in a period
\item \texttt{get\_market\_status} — NYSE / NASDAQ open/closed
\item \texttt{get\_news\_sentiment} — Alpha Vantage news + sentiment
\item \texttt{get\_company\_overview} — P/E, EPS, market cap, 52-wk range
\item \texttt{get\_tickers\_by\_sector} — SQL on local S\&P 500 DB
\item \texttt{query\_local\_db} — free-form SQL
\end{itemize}
```

---

## Eugenio's Multi-Agent: Adaptive Router-Verifier

\flowpoint{Router classifies each question into domains (market / fundamentals / news) and selects only needed specialists.}

\flowpoint{Execution mode: parallel when independent, staged when one agent needs another's output first.}

\flowpoint{Evidence Verifier mechanically scores each specialist: tool usage, schema compliance, answer substance $\to$ confidence 0--1. No extra LLM call.}

\flowpoint{Aggregator merges results sorted by confidence --- higher-evidence sources win on conflicts.}

```tex
\vspace{0.3em}
\centering
\scriptsize
\begin{tabular}{ll}
\hline
\textbf{Path} & \textbf{LLM Calls} \\
\hline
Router + 1 specialist + Aggregator & 3 \\
Router + 3 parallel specialists + Aggregator & 5 \\
\hline
\end{tabular}
```

---

## Oguz's Multi-Agent: Orchestrator-Critic-Synthesizer

\flowpoint{Orchestrator analyzes question, selects specialists, and detects ranking dependencies via structured JSON.}

\flowpoint{Two execution modes: single-phase (ThreadPoolExecutor) or two-phase when Phase 1 output feeds Phase 2.}

\flowpoint{LLM Critic (1 call per specialist) checks internal consistency --- do numbers in the answer match the raw tool output?}

\flowpoint{Synthesizer merges results: table format for multi-ticker comparisons, prose for others.}

```tex
\vspace{0.3em}
\centering
\scriptsize
\begin{tabular}{ll}
\hline
\textbf{Path} & \textbf{LLM Calls} \\
\hline
Orch. + 1 specialist + 1 Critic + Synth. & 4 \\
Orch. + 3 specialists + 3 Critics + Synth. & 8 \\
Two-phase + 3 specialists + 3 Critics + Synth. & 8 \\
\hline
\end{tabular}
```

---

## Architecture Comparison

```tex
\renewcommand{\arraystretch}{1.3}
\centering
\small
\begin{tabular}{lcc}
\hline
 & \textbf{Eugenio (Router-Verifier)} & \textbf{Oguz (Orch.-Critic-Synth.)} \\
\hline
Routing       & LLM Router              & LLM Orchestrator \\
Verification  & Mechanical scoring       & LLM Critic per agent \\
Aggregation   & Confidence-sorted merge  & Format-aware Synthesizer \\
Dependencies  & Parallel vs Staged       & Single-phase vs Two-phase \\
LLM calls     & 3--5                     & 4--8 \\
\hline
\end{tabular}
```

---

## Results --- gpt-4o-mini

```tex
\begin{columns}[T]
\begin{column}{0.56\textwidth}
\centering
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{
\begin{tabular}{l|rrrr|rrrr}
\hline
 & \multicolumn{4}{c|}{\textbf{Eugenio}} & \multicolumn{4}{c}{\textbf{Oguz}} \\
\textbf{Arch.} & \textbf{E} & \textbf{M} & \textbf{H} & \textbf{All} & \textbf{E} & \textbf{M} & \textbf{H} & \textbf{All} \\
\hline
Baseline & 26.7 &  0.0 &  0.0 &  8.9 &  0.0 &  0.0 &  0.0 &  0.0 \\
Single   & 80.0 & 86.7 & 53.3 & 73.3 & 60.0 & 86.7 & 46.7 & 64.4 \\
\textcolor{accent2}{Multi} & \textcolor{accent2}{93.3} & \textcolor{accent2}{86.7} & \textcolor{accent2}{60.0} & \textcolor{accent2}{80.0} & 60.0 & 73.3 & 46.7 & 60.0 \\
\hline
\end{tabular}
}
\end{column}
\begin{column}{0.40\textwidth}
\small
\uncover<2->{
\textcolor{accent2}{Eugenio's Multi-Agent wins} at 80\% overall --- best result across all runs.
}

\vspace{0.4em}
\uncover<3->{
Both Single Agents outperform baselines by 60+ pts.
}

\vspace{0.4em}
\uncover<4->{
Oguz's Single (64.4\%) edges out his Multi (60\%) --- Critic overhead without gain.
}
\end{column}
\end{columns}
```

---

## Results --- gpt-4o

```tex
\begin{columns}[T]
\begin{column}{0.56\textwidth}
\centering
\scriptsize
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{
\begin{tabular}{l|rrrr|rrrr}
\hline
 & \multicolumn{4}{c|}{\textbf{Eugenio}} & \multicolumn{4}{c}{\textbf{Oguz}} \\
\textbf{Arch.} & \textbf{E} & \textbf{M} & \textbf{H} & \textbf{All} & \textbf{E} & \textbf{M} & \textbf{H} & \textbf{All} \\
\hline
Baseline &  26.7 &  0.0 &  0.0 &  8.9 &  0.0 &  0.0 &  0.0 &  0.0 \\
Single   &  73.3 & 86.7 & 40.0 & 66.7 & 60.0 & 93.3 & 66.7 & 73.3 \\
Multi    &  80.0 & 80.0 & 40.0 & 66.7 & \textcolor{accent2}{66.7} & \textcolor{accent2}{86.7} & \textcolor{accent2}{66.7} & \textcolor{accent2}{74.4} \\
\hline
\end{tabular}
}
\end{column}
\begin{column}{0.40\textwidth}
\small
\uncover<2->{
\textcolor{accent2}{Oguz's Multi-Agent wins on gpt-4o} at 74.4\% --- LLM Critic pays off with the stronger model.
}

\vspace{0.4em}
\uncover<3->{
Eugenio's Single and Multi tie at 66.7\% --- but Multi is 2$\times$ slower (16.7s vs 7.8s).
}

\vspace{0.4em}
\uncover<4->{
Hard questions remain the bottleneck: 40--67\% across all runs.
}
\end{column}
\end{columns}
```

---

## Results --- Timing and Hallucinations

```tex
\centering
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\small
\resizebox{0.9\linewidth}{!}{
\begin{tabular}{ll|rr|rr}
\hline
 & & \multicolumn{2}{c|}{\textbf{Avg Time (s)}} & \multicolumn{2}{c}{\textbf{Hallucinations}} \\
\textbf{Arch.} & \textbf{Model} & \textbf{Eugenio} & \textbf{Oguz} & \textbf{Eugenio} & \textbf{Oguz} \\
\hline
Baseline & 4o-mini & 2.8  & 3.4  & 0 & 0 \\
Single   & 4o-mini & 5.8  & 8.8  & 1 & 0 \\
Multi    & 4o-mini & 11.2 & 16.4 & 2 & 0 \\
\hline
Baseline & 4o      & 1.5  & 3.1  & 0 & 0 \\
Single   & 4o      & 7.8  & 7.4  & 0 & 0 \\
Multi    & 4o      & 16.7 & 7.0  & 3 & 0 \\
\hline
\end{tabular}
}
```

---

## Insights --- What Worked

\flowpoint{Tools are essential: baselines scored 0--8.9\% while tool-equipped agents reached 60--80\%.}

\flowpoint{Eugenio's mechanical Verifier kept latency low (5.8s single, 11.2s multi) while achieving the top score (80\%).}

\flowpoint{Oguz's LLM Critic produced zero hallucinations across all runs --- the extra LLM call for consistency checking paid off.}

\flowpoint{Both staged/two-phase execution modes correctly handled cross-domain chains (sector lookup $\to$ price $\to$ fundamentals).}

---

## Insights --- What Failed and Lessons Learned

\flowpoint{Multi-Agent added 2$\times$ latency in every configuration. On gpt-4o-mini, Oguz's Multi (60\%) was slower than his Single (64.4\%) and less accurate.}

\flowpoint{Hard questions (multi-condition, cross-domain) remained the bottleneck: 40--67\% even in the best runs. Tool chaining errors accumulate.}

\flowpoint{Eugenio's mechanical verifier was cheaper but let through 1--3 hallucinations; Oguz's LLM Critic caught them all but cost extra API calls.}

\flowpoint{Complexity must earn its cost: multi-agent only outperformed single-agent in 2 of 4 runs (Eugenio 4o-mini, Oguz 4o).}

---

<!-- center -->
{\large\color{accent} Demo --- Streamlit App\par}
\vspace{0.5em}
{\color{muted} Agent selector \textbullet\ Model selector \textbullet\ 3-turn follow-up \textbullet\ Context memory\par}
<!-- /center -->
