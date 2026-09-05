"""
app.py
======
Entry point for the WhatsApp Chat Analyzer Streamlit dashboard.

This file is intentionally kept "thin": it wires together the sidebar
(upload + filters), the section navigation, and calls into helper.py
(statistics), charts.py (Plotly figures), and utils.py (filters, KPIs,
search, export). No data-crunching logic lives here directly.

Part 4 — True lazy section loading
------------------------------------
Previously every analytics feature lived inside a `with tab:` block
under `st.tabs()`. Streamlit executes EVERY tab body on EVERY rerun
regardless of which tab is visible, so switching to "Overview" was
silently re-running Sentiment, Topics, Toxicity, and the AI/RAG stack
too. See nav.py for the full explanation.

The fix: navigation is now a plain single-select control (see
`nav.top_nav` / `nav.sub_nav`), and the whole layout is a chain of
`if/elif` branches, each calling exactly one `render_*()` function.
Only the branch matching the current selection executes — everything
else is skipped entirely, not just visually hidden.

Run with:
    streamlit run app.py
"""
import base64
import contextlib
import hashlib
import html
import re
import time

import pandas as pd
import streamlit as st

import ai_helper
import analytics
import charts
import config
import data_layer
import helper
import nav
import nlp_helper
import sentiment
import topics
import toxicity
import utils
import vector_store

# ---------------------------------------------------------------------------
# SHADOWTRACE design system
# ---------------------------------------------------------------------------
# Pure presentation layer: CSS only, injected on top of whatever
# utils.inject_custom_css() already does. Nothing here touches data,
# caching, navigation logic, or reruns — it only restyles the DOM
# Streamlit already renders. Safe to delete without affecting any
# functionality.
_SHADOWTRACE_CSS = """
<style>
:root{
  --st-bg:#0a0a0f;
  --st-surface:#14141c;
  --st-surface-2:#1b1b26;
  --st-border:rgba(255,255,255,0.08);
  --st-accent:#7c5cff;
  --st-accent-2:#22d3ee;
  --st-text:#f5f5f7;
  --st-text-muted:#9a9aab;
  --st-success:#34d399;
  --st-warning:#fbbf24;
  --st-error:#f87171;
}

.stApp{
  background:
    radial-gradient(circle at 12% 0%, rgba(124,92,255,0.10), transparent 40%),
    radial-gradient(circle at 88% 12%, rgba(34,211,238,0.06), transparent 38%),
    var(--st-bg);
}
.block-container{ padding-top: 1.5rem; max-width: 1240px; }

/* Top loading bar — a real "something is happening" indicator that
   sweeps left-to-right whenever Streamlit is running (any rerun:
   switching sections/tabs, changing filters, uploading a file). It
   does not claim a fake numeric percentage; it's an indeterminate
   progress cue, the same convention as a page-load bar (YouTube/
   GitHub-style), driven purely by CSS from Streamlit's own native
   "running" indicator (`[data-testid="stStatusWidget"]`) via :has() —
   no extra reruns, no JS polling. */
.st-top-loading-bar{
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 3px;
  z-index: 999999;
  background: linear-gradient(90deg, transparent, var(--st-accent) 35%, var(--st-accent-2) 65%, transparent);
  background-size: 250% 100%;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s ease;
}
body:has([data-testid="stStatusWidget"]) .st-top-loading-bar{
  opacity: 1;
  animation: st-loading-sweep 1s linear infinite;
}
@keyframes st-loading-sweep{
  0%{ background-position: 200% 0; }
  100%{ background-position: -50% 0; }
}

/* Inline loading bar — used inside a section/card while its own data
   is being computed (see `_loading_bar()`), so a tab switch shows an
   obvious "loading" placeholder instead of the previous tab's stale
   cards lingering on screen mid-rerun. */
.st-loading-bar-wrap{
  padding: 1.75rem 0.25rem;
  animation: st-fade-in 0.2s ease;
}
.st-loading-bar-track{
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: var(--st-surface-2);
  overflow: hidden;
  border: 1px solid var(--st-border);
}
.st-loading-bar-fill{
  height: 100%;
  width: 40%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  animation: st-loading-fill 1.15s ease-in-out infinite;
}
@keyframes st-loading-fill{
  0%{ transform: translateX(-100%); }
  100%{ transform: translateX(350%); }
}
.st-loading-bar-label{
  margin-top: 0.6rem;
  color: var(--st-text-muted);
  font-size: 0.85rem;
  text-align: center;
}

/* Hero header */
.st-hero{
  padding: 1.75rem 0 1.25rem 0;
  border-bottom: 1px solid var(--st-border);
  margin-bottom: 1.25rem;
  animation: st-fade-in 0.5s ease;
}
.st-hero-brand{
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
/* Gradient-diamond brand mark, reused wherever the wordmark appears
   (sidebar, landing hero, loading screen, footer) so the product has
   one consistent visual identity — same asset as the favicon. */
.st-brand-logo{
  display: inline-block;
  vertical-align: middle;
  margin-right: 0.5rem;
  filter: drop-shadow(0 0 10px rgba(124, 92, 255, 0.45));
}
.st-brand-row{
  display: flex;
  align-items: center;
  gap: 0.1rem;
}
.st-landing-brand.st-brand-row,
.st-analyzing-brand.st-brand-row,
.st-footer-brand.st-brand-row{
  justify-content: center;
}
.st-hero-tagline{
  color: var(--st-text-muted);
  font-size: 0.95rem;
  margin-top: 0.15rem;
}

/* Onboarding / empty (upload/landing) state — cinematic but restrained:
   a soft violet/cyan glow field standing in for the reference's
   "conversation intelligence" background art, built entirely from CSS
   gradients so it costs nothing to render and never fights Streamlit's
   DOM. */
.st-empty-state{
  position: relative;
  overflow: hidden;
  text-align:center;
  padding: 4.5rem 1.5rem;
  border: 1px solid var(--st-border);
  border-radius: 20px;
  background:
    radial-gradient(circle at 50% -10%, rgba(124,92,255,0.16), transparent 55%),
    radial-gradient(circle at 85% 90%, rgba(34,211,238,0.10), transparent 45%),
    radial-gradient(circle at 8% 85%, rgba(124,92,255,0.08), transparent 40%),
    var(--st-surface);
  animation: st-fade-in 0.6s ease;
}
.st-empty-state::before{
  content:"";
  position:absolute; inset:0;
  background-image:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(ellipse at 50% 30%, black 0%, transparent 72%);
  pointer-events: none;
}
.st-empty-content{ position: relative; z-index: 1; }
.st-empty-eyebrow{
  display:inline-flex; align-items:center; gap:0.4rem;
  color: var(--st-accent-2); font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase;
  padding: 0.25rem 0.75rem; border: 1px solid rgba(34,211,238,0.3);
  border-radius: 999px; background: rgba(34,211,238,0.06);
  margin-bottom: 1.1rem;
}
.st-empty-brand{
  font-size: 3rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  background: linear-gradient(90deg, #ffffff, var(--st-accent-2) 65%, var(--st-accent));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.st-empty-tagline{ color: var(--st-text-muted); margin-top: 0.4rem; font-size: 1.05rem; }
.st-empty-body{ color: var(--st-text-muted); max-width: 520px; margin: 1.25rem auto 0 auto; line-height: 1.6; }

/* Cards */
.st-card{
  background: var(--st-surface);
  border: 1px solid var(--st-border);
  border-radius: 14px;
  padding: 1.1rem 1.2rem;
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.st-card:hover{ border-color: rgba(124,92,255,0.35); }

/* Native bordered containers (st.container(border=True)) restyled into
   the same "panel" language as the rest of the design system — this is
   a stable, documented Streamlit API rather than a private test id, so
   it's the preferred way to get card-wrapped charts/sections. */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-color: var(--st-border) !important;
  border-radius: 16px !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.015), transparent 40%), var(--st-surface);
  transition: border-color 0.15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{ border-color: rgba(124,92,255,0.28) !important; }

.st-panel-title{ font-size: 1.02rem; font-weight: 700; color: var(--st-text); }
.st-panel-subtitle{ color: var(--st-text-muted); font-size: 0.82rem; margin-top: 0.05rem; margin-bottom: 0.5rem; }

/* Insight / highlight list */
.st-insight-row{
  display:flex; align-items:flex-start; gap:0.65rem;
  padding: 0.55rem 0; border-bottom: 1px solid var(--st-border);
  font-size: 0.9rem; color: var(--st-text);
}
.st-insight-row:last-child{ border-bottom: none; }
.st-insight-mark{ color: var(--st-accent-2); flex-shrink:0; }

/* Streamlit metric restyle -> premium KPI card w/ subtle top-edge glow */
div[data-testid="stMetric"]{
  position: relative;
  background: linear-gradient(180deg, rgba(124,92,255,0.05), transparent 55%), var(--st-surface);
  border: 1px solid var(--st-border);
  border-radius: 14px;
  padding: 1rem 1.1rem 0.85rem 1.1rem;
  transition: border-color 0.15s ease, transform 0.15s ease;
  overflow: hidden;
}
div[data-testid="stMetric"]::before{
  content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  opacity: 0; transition: opacity 0.15s ease;
}
div[data-testid="stMetric"]:hover{ border-color: rgba(124,92,255,0.35); transform: translateY(-2px); }
div[data-testid="stMetric"]:hover::before{ opacity: 1; }
div[data-testid="stMetricValue"]{ color: var(--st-text); font-weight: 700; }
div[data-testid="stMetricLabel"]{ color: var(--st-text-muted); font-size: 0.82rem; }

/* Animated KPI card (Overview headline stats) — same visual identity
   as div[data-testid="stMetric"] above, plus a pure-CSS count-up for
   whole-number values via an animatable custom property. Chromium
   (Streamlit's frontend) supports @property; browsers that don't
   simply show the final number with no animation — never a missing
   or wrong value either way. */
@property --num{
  syntax: '<integer>';
  initial-value: 0;
  inherits: false;
}
.st-kpi-card{
  position: relative;
  background: linear-gradient(180deg, rgba(124,92,255,0.05), transparent 55%), var(--st-surface);
  border: 1px solid var(--st-border);
  border-radius: 14px;
  padding: 1rem 1.1rem 0.85rem 1.1rem;
  transition: border-color 0.15s ease, transform 0.15s ease;
  overflow: hidden;
}
.st-kpi-card::before{
  content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  opacity: 0; transition: opacity 0.15s ease;
}
.st-kpi-card:hover{ border-color: rgba(124,92,255,0.35); transform: translateY(-2px); }
.st-kpi-card:hover::before{ opacity: 1; }
.st-kpi-icon{ color: var(--st-accent-2); }
.st-kpi-label{ color: var(--st-text-muted); font-size: 0.82rem; margin-bottom: 0.2rem; }
.st-kpi-value{ color: var(--st-text); font-weight: 700; font-size: 1.9rem; line-height: 1.25; }
.st-kpi-value.st-kpi-count{
  --num: 0;
  counter-reset: st-kpi-num var(--num);
  animation: st-kpi-count-up 1.1s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.st-kpi-value.st-kpi-count::after{ content: counter(st-kpi-num); }
.st-kpi-value.st-kpi-static{ animation: st-fade-in 0.5s ease; }
@keyframes st-kpi-count-up{
  from{ --num: 0; }
}
@media (prefers-reduced-motion: reduce){
  .st-kpi-value.st-kpi-count{ animation: none; }
}

/* Segmented control / radio nav restyle -> pill nav */
div[data-testid="stSegmentedControl"] button,
div[role="radiogroup"] label{
  border-radius: 999px !important;
  transition: background 0.15s ease, color 0.15s ease;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"]{
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2)) !important;
  color: #0a0a0f !important;
  border: none !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--st-surface);
  border-right: 1px solid var(--st-border);
}

/* Buttons */
.stButton button, .stDownloadButton button{
  border-radius: 10px;
  border: 1px solid var(--st-border);
  transition: transform 0.1s ease, border-color 0.15s ease;
}
.stButton button:hover, .stDownloadButton button:hover{
  border-color: var(--st-accent);
  transform: translateY(-1px);
}
.stButton button[kind="primary"]{
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  border: none;
}

/* Footer */
.st-footer{
  margin-top: 3rem;
  padding-top: 1.25rem;
  border-top: 1px solid var(--st-border);
  text-align: center;
  color: var(--st-text-muted);
  font-size: 0.85rem;
}
.st-footer-brand{ color: var(--st-text); font-weight: 700; letter-spacing: 0.04em; }

/* Sidebar brand block */
.st-sidebar-brand{
  padding: 0.25rem 0 1rem 0;
  margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--st-border);
}
.st-sidebar-brand-word{
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  color: var(--st-text);
}
.st-sidebar-brand-word span{
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.st-sidebar-brand-tag{
  color: var(--st-text-muted);
  font-size: 0.75rem;
  margin-top: 0.1rem;
}

/* Top navbar — primary section switcher (Overview / Activity / People /
   Conversations / Insights / AI Assistant / Tools), rendered in the
   main content area via st.container(key="top_navbar"), not the
   sidebar. Styles both possible underlying widgets: st.segmented_control
   (preferred — renders as [data-testid="stButtonGroup"] wrapping a
   role="radiogroup" of <button role="radio"> elements, confirmed via a
   throwaway probe page's live DOM — NOT <label>/<input>, so selectors
   below target the real button markup) and the plain
   st.radio(horizontal=True) fallback (<label>/<input>) for older
   Streamlit versions. */
div.st-key-top_navbar{
  margin: 0 0 1.75rem 0; padding: 0.45rem; border-radius: 18px;
  background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01)), var(--st-surface);
  border: 1px solid var(--st-border);
  box-shadow: 0 10px 30px -18px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04);
  animation: st-fade-in 0.4s ease;
}
div.st-key-top_navbar [data-testid="stButtonGroup"]{
  flex-wrap: wrap; gap: 0.35rem;
}
div.st-key-top_navbar [data-testid="stButtonGroup"] [role="radiogroup"]{
  flex-wrap: wrap; gap: 0.35rem; width: 100%;
}
div.st-key-top_navbar [data-testid="stButtonGroup"] button{
  border-radius: 12px !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  font-weight: 600;
  letter-spacing: 0.01em;
  padding: 0.6rem 1.15rem !important;
  color: var(--st-text-muted);
  transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease, transform 0.18s ease, box-shadow 0.18s ease;
}
div.st-key-top_navbar [data-testid="stButtonGroup"] button:hover{
  background: rgba(255,255,255,0.06) !important;
  color: var(--st-text) !important;
  transform: translateY(-1px);
}
div.st-key-top_navbar [data-testid="stButtonGroup"] button[aria-checked="true"],
div.st-key-top_navbar [data-testid="stButtonGroup"] button[data-selected="true"]{
  background: linear-gradient(135deg, rgba(124,92,255,0.32), rgba(34,211,238,0.14)) !important;
  border-color: rgba(124,92,255,0.55) !important;
  box-shadow: 0 0 0 1px rgba(124,92,255,0.25), 0 10px 22px -10px rgba(124,92,255,0.6);
  color: var(--st-text) !important;
  font-weight: 700 !important;
  transform: translateY(-1px);
}
div.st-key-top_navbar [data-testid="stButtonGroup"] button[aria-checked="true"] p,
div.st-key-top_navbar [data-testid="stButtonGroup"] button[data-selected="true"] p{
  color: var(--st-text) !important;
  font-weight: 700 !important;
}
/* Radio fallback (older Streamlit without segmented_control) */
div.st-key-top_navbar div[role="radiogroup"] label{
  border-radius: 12px; padding: 0.6rem 1.15rem; font-weight: 600;
  border: 1px solid transparent;
  transition: background 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}
div.st-key-top_navbar div[role="radiogroup"] label:hover{
  background: rgba(255,255,255,0.06);
  transform: translateY(-1px);
}
div.st-key-top_navbar div[role="radiogroup"] label:has(input:checked){
  background: linear-gradient(135deg, rgba(124,92,255,0.32), rgba(34,211,238,0.14));
  border-color: rgba(124,92,255,0.55);
  box-shadow: 0 0 0 1px rgba(124,92,255,0.25), 0 10px 22px -10px rgba(124,92,255,0.6);
}
div.st-key-top_navbar div[role="radiogroup"] label:has(input:checked) p{
  color: var(--st-text) !important; font-weight: 700;
}

/* In-section sub-navigation tabs (e.g. Insights: Sentiment / Topics /
   Emotions / Toxicity / Search) — a soft rounded "chip strip", deliberately
   lighter and smaller than the main pill navbar above so the two nav
   levels read as distinct hierarchy, not duplicated navigation. */
div[class*="st-key-subnav_"]{
  margin: 0.35rem 0 1.5rem 0;
  padding: 0.3rem;
  border-radius: 13px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.06);
}
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"],
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"] [role="radiogroup"],
div[class*="st-key-subnav_"] div[role="radiogroup"]{
  flex-wrap: wrap; gap: 0.2rem;
}
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"] button,
div[class*="st-key-subnav_"] div[role="radiogroup"] label{
  border-radius: 9px !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  padding: 0.4rem 0.9rem !important;
  font-size: 0.86rem;
  color: var(--st-text-muted);
  transition: color 0.15s ease, background 0.15s ease, border-color 0.15s ease;
}
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"] button:hover,
div[class*="st-key-subnav_"] div[role="radiogroup"] label:hover{
  color: var(--st-text);
  background: rgba(255,255,255,0.05) !important;
}
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"] button[aria-checked="true"],
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"] button[data-selected="true"],
div[class*="st-key-subnav_"] div[role="radiogroup"] label:has(input:checked){
  background: rgba(124,92,255,0.16) !important;
  border-color: rgba(124,92,255,0.4) !important;
  box-shadow: 0 4px 12px -6px rgba(124,92,255,0.45);
}
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"] button[aria-checked="true"] p,
div[class*="st-key-subnav_"] [data-testid="stButtonGroup"] button[data-selected="true"] p,
div[class*="st-key-subnav_"] div[role="radiogroup"] label:has(input:checked) p{
  color: var(--st-accent-2) !important; font-weight: 700;
}

/* Page headers */
.st-page-header{ margin-bottom: 1rem; animation: st-fade-in 0.3s ease; }
.st-page-header .st-page-title{ font-size: 1.4rem; font-weight: 700; color: var(--st-text); }
.st-page-header .st-page-subtitle{ color: var(--st-text-muted); font-size: 0.92rem; margin-top: 0.15rem; }

.st-page-hero{
  margin-bottom: 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--st-border);
  animation: st-fade-in 0.4s ease;
}
.st-page-hero .st-page-title{
  font-size: 1.9rem;
  font-weight: 800;
  background: linear-gradient(90deg, var(--st-text), var(--st-text) 60%, var(--st-accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.st-page-hero .st-page-subtitle{ color: var(--st-text-muted); font-size: 1rem; margin-top: 0.2rem; }

/* Main-content brand strip — sits once above the top navbar, on every
   dashboard section, so the product identity (wordmark + tagline)
   stays visible in the main content area itself, not just the
   sidebar. Deliberately compact and muted (small logo, thin divider)
   so it reads as a header lockup, not a second competing headline —
   the page's own title/subtitle (see .st-page-header above) remains
   the one clear heading per section. */
.st-main-header-brand{
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.15rem;
  padding-bottom: 0.9rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--st-border);
  animation: st-fade-in 0.35s ease;
}
.st-main-header-brand .st-brand-row{ gap: 0.35rem; }
.st-main-header-brand-word{
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: 0.03em;
  color: var(--st-text);
  white-space: nowrap;
}
.st-main-header-brand-word span{
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.st-main-header-brand-tag{
  color: var(--st-text-muted);
  font-size: 0.85rem;
  font-style: normal;
  white-space: nowrap;
  margin-left: 0.1rem;
}
@media (max-width: 640px){
  .st-main-header-brand-word{ font-size: 1.25rem; }
  .st-main-header-brand-tag{ font-size: 0.75rem; }
}

/* AI chat bubbles */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"]{
  background: linear-gradient(135deg, rgba(124,92,255,0.14), rgba(34,211,238,0.07));
  border: 1px solid rgba(124,92,255,0.25);
  border-radius: 14px;
  padding: 0.35rem 0.15rem;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"]{
  background: var(--st-surface-2);
  border: 1px solid var(--st-border);
  border-radius: 14px;
  padding: 0.35rem 0.15rem;
}

/* Suggested-question chips (plain buttons restyled as pills) */
div.st-key-chip_row .stButton button{
  border-radius: 999px;
  font-size: 0.82rem;
  padding: 0.35rem 0.9rem;
}

/* Chat composer glow-on-focus (targets the standard chat input testid,
   stable across Streamlit versions) */
div[data-testid="stChatInput"]{
  border-radius: 14px;
  border: 1px solid var(--st-border);
  background: var(--st-surface);
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stChatInput"]:focus-within{
  border-color: var(--st-accent);
  box-shadow: 0 0 0 3px rgba(124,92,255,0.15);
}

/* File uploader dropzone — restyle the default Streamlit widget into a
   premium upload panel without replacing its markup (keeps drag/drop,
   click-to-browse, and validation fully intact). Selector covers both
   older and newer Streamlit internal structures; harmless no-op if a
   given version doesn't match one of them. */
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section{
  background: linear-gradient(180deg, rgba(124,92,255,0.06), transparent 60%), var(--st-surface-2) !important;
  border: 1.5px dashed rgba(124,92,255,0.35) !important;
  border-radius: 14px !important;
  transition: border-color 0.15s ease, background 0.15s ease;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover{
  border-color: var(--st-accent) !important;
}

@keyframes st-fade-in{
  from{ opacity: 0; transform: translateY(4px); }
  to{ opacity: 1; transform: translateY(0); }
}
[data-testid="stVerticalBlock"] > div{ animation: st-fade-in 0.25s ease; }

/* Chart entrance animation — every Plotly figure fades + settles into
   place as soon as it renders (line/area/bar/donut/scatter data itself
   is untouched; this only animates the container it sits in, so it
   costs nothing extra to compute and never depends on chart type). */
[data-testid="stPlotlyChart"]{
  animation: st-chart-in 0.65s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes st-chart-in{
  from{ opacity: 0; transform: translateY(12px) scale(0.98); }
  to{ opacity: 1; transform: translateY(0) scale(1); }
}
@media (prefers-reduced-motion: reduce){
  [data-testid="stPlotlyChart"]{ animation: none; }
}

/* ===========================================================
   STATE 1 — Landing page (no sidebar). Scoped under .st-landing
   so none of it leaks into the STATE 2 dashboard. Background is
   pure CSS gradients/keyframes -- no JS loop, no Streamlit rerun
   is ever triggered by these animations.
   =========================================================== */
.st-landing{ position: relative; }

.st-landing-bg{
  position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none;
}
.st-landing-bg::before, .st-landing-bg::after{
  content:""; position:absolute; border-radius:50%; filter: blur(60px); opacity:0.55;
}
.st-landing-bg::before{
  width: 46vw; height: 46vw; left: -12vw; top: -14vw;
  background: radial-gradient(circle, rgba(124,92,255,0.35), transparent 70%);
  animation: st-drift-a 22s ease-in-out infinite alternate;
}
.st-landing-bg::after{
  width: 40vw; height: 40vw; right: -10vw; bottom: -12vw;
  background: radial-gradient(circle, rgba(34,211,238,0.28), transparent 70%);
  animation: st-drift-b 26s ease-in-out infinite alternate;
}
.st-landing-particles{
  position:absolute; inset:0;
  background-image:
    radial-gradient(2px 2px at 12% 22%, rgba(255,255,255,0.35), transparent 60%),
    radial-gradient(2px 2px at 82% 15%, rgba(124,92,255,0.55), transparent 60%),
    radial-gradient(1.5px 1.5px at 35% 68%, rgba(34,211,238,0.55), transparent 60%),
    radial-gradient(1.5px 1.5px at 68% 78%, rgba(255,255,255,0.28), transparent 60%),
    radial-gradient(2px 2px at 92% 55%, rgba(124,92,255,0.45), transparent 60%),
    radial-gradient(1.5px 1.5px at 20% 85%, rgba(34,211,238,0.4), transparent 60%);
  animation: st-drift-c 18s ease-in-out infinite alternate;
}
.st-landing-grid{
  position:absolute; inset:0;
  background-image:
    linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(ellipse at 50% 25%, black 0%, transparent 70%);
  -webkit-mask-image: radial-gradient(ellipse at 50% 25%, black 0%, transparent 70%);
}
@keyframes st-drift-a{ from{ transform: translate(0,0); } to{ transform: translate(3vw, 4vw); } }
@keyframes st-drift-b{ from{ transform: translate(0,0); } to{ transform: translate(-3vw, -4vw); } }
@keyframes st-drift-c{ from{ transform: translate(0,0) scale(1); } to{ transform: translate(-1.5vw, 2vw) scale(1.05); } }
@media (prefers-reduced-motion: reduce){
  .st-landing-bg::before, .st-landing-bg::after, .st-landing-particles{ animation: none !important; }
}

.st-landing-hero{ text-align:center; padding: 3rem 1rem 0.5rem 1rem; animation: st-fade-in 0.6s ease; }
.st-landing-eyebrow{
  display:inline-flex; align-items:center; gap:0.4rem;
  color: var(--st-accent-2); font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase;
  padding: 0.25rem 0.75rem; border: 1px solid rgba(34,211,238,0.3);
  border-radius: 999px; background: rgba(34,211,238,0.06);
  margin-bottom: 1.1rem;
}
.st-landing-brand{
  font-size: 3.2rem; font-weight: 800; letter-spacing: 0.1em;
  background: linear-gradient(90deg, #ffffff, var(--st-accent-2) 65%, var(--st-accent));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.st-landing-tagline{ color: var(--st-text-muted); font-size: 1.1rem; margin-top: 0.35rem; }
.st-landing-desc{ color: var(--st-text-muted); max-width: 560px; margin: 1.1rem auto 0 auto; line-height: 1.6; }

/* Upload card — restyles the real st.file_uploader; the widget itself
   is untouched (still real drag/drop + click-to-browse + validation).
   Applied via st.container(key="upload_card"), which Streamlit renders
   as a genuine wrapper div carrying the "st-key-upload_card" class —
   this is what makes the descendant selectors below actually match
   (a plain st.markdown('<div>...') / st.markdown('</div>') pair does
   NOT nest the widgets in between; they stay siblings in the DOM). */
div.st-key-upload_card{
  max-width: 640px; margin: 2rem auto 0 auto;
  position: relative;
  border: 1px solid var(--st-border); border-radius: 22px;
  padding: 2.1rem 2rem 1.5rem 2rem;
  background:
    radial-gradient(circle at 50% 0%, rgba(124,92,255,0.14), transparent 60%),
    linear-gradient(180deg, rgba(124,92,255,0.06), transparent 55%),
    var(--st-surface);
  box-shadow: 0 20px 50px -25px rgba(0,0,0,0.65);
  text-align:center; animation: st-fade-in 0.5s ease;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}
div.st-key-upload_card:hover{
  border-color: rgba(124,92,255,0.35);
  box-shadow: 0 24px 60px -22px rgba(124,92,255,0.28);
  transform: translateY(-1px);
}
/* Thin gradient accent rule across the top of the card */
div.st-key-upload_card::before{
  content: ""; position: absolute; top: 0; left: 14%; right: 14%; height: 2px;
  background: linear-gradient(90deg, transparent, var(--st-accent), var(--st-accent-2), transparent);
  border-radius: 999px; opacity: 0.75;
}
.st-upload-icon-wrap{
  width: 52px; height: 52px; margin: 0 auto 0.9rem auto;
  display: flex; align-items: center; justify-content: center;
  border-radius: 14px;
  background: linear-gradient(135deg, var(--st-accent), var(--st-accent-2));
  box-shadow: 0 8px 24px -8px rgba(124,92,255,0.55);
}
.st-upload-icon-wrap svg{ width: 24px; height: 24px; stroke: #fff; }
.st-upload-title{ font-size:1.2rem; font-weight:700; color: var(--st-text); letter-spacing: 0.01em; }
.st-upload-sub{ color: var(--st-text-muted); font-size:0.9rem; margin: 0.3rem 0 1.25rem 0; }
.st-upload-hint{
  color: var(--st-text-muted); font-size: 0.78rem; margin-top: 0.85rem;
  display:flex; align-items:center; justify-content:center; gap:0.4rem;
}
.st-upload-hint::before, .st-upload-hint::after{
  content:""; flex:1; max-width:64px; height:1px; background: var(--st-border);
}

div.st-key-upload_card [data-testid="stFileUploaderDropzone"],
div.st-key-upload_card [data-testid="stFileUploader"] section{
  background: linear-gradient(180deg, rgba(124,92,255,0.08), transparent 65%), var(--st-surface-2) !important;
  border: 1.5px dashed rgba(124,92,255,0.4) !important;
  border-radius: 16px !important;
  padding: 1.1rem !important;
  transition: border-color 0.2s ease, background 0.2s ease;
}
div.st-key-upload_card [data-testid="stFileUploaderDropzone"]:hover,
div.st-key-upload_card [data-testid="stFileUploader"] section:hover{
  border-color: var(--st-accent) !important;
  background: linear-gradient(180deg, rgba(124,92,255,0.14), transparent 65%), var(--st-surface-2) !important;
}
/* The built-in "Browse files" button — recolor to match the brand
   instead of Streamlit's default red/orange accent. */
div.st-key-upload_card [data-testid="stFileUploaderDropzone"] button,
div.st-key-upload_card [data-testid="stFileUploader"] section button{
  background: var(--st-surface) !important;
  border: 1px solid rgba(124,92,255,0.4) !important;
  color: var(--st-text) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  transition: border-color 0.15s ease, color 0.15s ease;
}
div.st-key-upload_card [data-testid="stFileUploaderDropzone"] button:hover,
div.st-key-upload_card [data-testid="stFileUploader"] section button:hover{
  border-color: var(--st-accent) !important;
  color: var(--st-accent-2) !important;
}
/* Uploaded-file chip shown once a file is selected */
div.st-key-upload_card [data-testid="stFileUploaderFile"]{
  background: var(--st-surface) !important;
  border: 1px solid var(--st-border) !important;
  border-radius: 10px !important;
}

/* Ready banner + filter panel (revealed only after a file is present) */
.st-ready-banner{
  max-width: 640px; margin: 1.5rem auto 0 auto; text-align:center;
  animation: st-fade-in 0.4s ease;
}
.st-ready-eyebrow{
  display:inline-flex; align-items:center; gap:0.4rem;
  color: var(--st-success); font-weight:700; font-size:0.85rem; letter-spacing:0.03em;
  padding: 0.35rem 0.9rem; border-radius: 999px;
  background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.25);
}
div.st-key-filter_panel{ max-width: 760px; margin: 1.1rem auto 0 auto; animation: st-fade-in 0.5s ease; }
div.st-key-filter_panel [data-testid="stExpander"],
div.st-key-filter_panel div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius: 18px !important;
}

/* Analyzing / loading screen (STATE transition) — indeterminate,
   non-fake progress: a scanning glow + a pulsing checklist. */
.st-analyzing{ text-align:center; padding: 4.5rem 1rem 3rem 1rem; animation: st-fade-in 0.4s ease; }
.st-analyzing-brand{
  font-size:1.6rem; font-weight:800; letter-spacing:0.08em;
  background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.st-analyzing-title{ color: var(--st-text); font-size:1.15rem; margin-top:0.6rem; font-weight:600; }
.st-analyzing-list{ list-style:none; padding:0; margin: 1.75rem auto 0 auto; max-width:320px; text-align:left; }
.st-analyzing-list li{
  color: var(--st-text-muted); padding:0.35rem 0; font-size:0.92rem;
  display:flex; align-items:center; gap:0.6rem;
}
.st-analyzing-list li::before{
  content:"◉"; color: var(--st-accent-2); flex-shrink:0;
  animation: st-pulse 1.4s ease-in-out infinite;
}
.st-analyzing-list li:nth-child(2)::before{ animation-delay: 0.2s; }
.st-analyzing-list li:nth-child(3)::before{ animation-delay: 0.4s; }
.st-analyzing-list li:nth-child(4)::before{ animation-delay: 0.6s; }
.st-analyzing-list li:nth-child(5)::before{ animation-delay: 0.8s; }
.st-analyzing-bar{
  width: 260px; height:4px; margin: 2rem auto 0 auto; border-radius:999px;
  background: rgba(255,255,255,0.08); overflow:hidden; position:relative;
}
.st-analyzing-bar::after{
  content:""; position:absolute; top:0; left:-40%; width:40%; height:100%;
  background: linear-gradient(90deg, transparent, var(--st-accent), var(--st-accent-2), transparent);
  animation: st-scan 1.3s ease-in-out infinite;
}
@keyframes st-scan{ 0%{ left:-40%; } 100%{ left:100%; } }
@keyframes st-pulse{ 0%,100%{ opacity:0.45; } 50%{ opacity:1; } }

/* Dashboard sidebar — "current analysis" summary card */
.st-current-analysis{
  margin: 0.25rem 0 0.6rem 0; padding: 0.7rem 0.8rem; border-radius: 12px;
  border: 1px solid var(--st-border); background: var(--st-surface-2);
}
.st-current-analysis-label{
  color: var(--st-text-muted); font-size:0.7rem; font-weight:700;
  letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.2rem;
}
.st-current-analysis-file{ color: var(--st-text); font-size:0.85rem; font-weight:600; word-break: break-word; }
.st-current-analysis-meta{ color: var(--st-text-muted); font-size:0.78rem; margin-top:0.15rem; }

/* People page — avatar-initial participant cards */
.st-avatar-card{
  display:flex; align-items:center; gap:0.75rem;
  padding: 0.6rem 0.7rem; margin-bottom: 0.5rem;
  border: 1px solid var(--st-border); border-radius: 12px; background: var(--st-surface-2);
}
.st-avatar-circle{
  width:38px; height:38px; border-radius:50%; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  font-weight:700; font-size:0.85rem; color:#0a0a0f;
  background: linear-gradient(135deg, var(--st-accent), var(--st-accent-2));
}
.st-avatar-body{ flex:1; min-width:0; }
.st-avatar-name{ color: var(--st-text); font-weight:600; font-size:0.88rem; }
.st-avatar-meta{ color: var(--st-text-muted); font-size:0.76rem; }
.st-avatar-bar{ height:5px; border-radius:999px; background: rgba(255,255,255,0.08); margin-top:0.3rem; overflow:hidden; }
.st-avatar-bar-fill{ height:100%; border-radius:999px; background: linear-gradient(90deg, var(--st-accent), var(--st-accent-2)); }

/* Responsive */
@media (max-width: 640px){
  .st-hero-brand{ font-size: 1.6rem; }
  .st-empty-brand{ font-size: 2rem; }
  .st-empty-state{ padding: 2.5rem 1rem; }
  .st-empty-body{ font-size: 0.92rem; }
  .st-page-hero .st-page-title{ font-size: 1.5rem; }
  .block-container{ padding-left: 0.75rem; padding-right: 0.75rem; }
  div[data-testid="stMetric"]{ padding: 0.75rem 0.85rem 0.65rem 0.85rem; }
  .st-landing-brand{ font-size: 2.1rem; }
  .st-landing-hero{ padding: 2rem 0.75rem 0.25rem 0.75rem; }
  div.st-key-upload_card{ padding: 1.5rem 1.1rem 1.1rem 1.1rem; }
  .st-analyzing{ padding: 3rem 0.75rem 2rem 0.75rem; }
}
</style>
"""


def _apply_shadowtrace_theme() -> None:
    """Inject the SHADOWTRACE visual theme. CSS-only; no reruns, no
    data access, no change to any widget's behavior or value."""
    st.markdown(_SHADOWTRACE_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str:
    """Base64-embed the gradient-diamond brand mark once so every brand
    lockup (sidebar, landing hero, loading screen) can reuse the exact
    same image inline, without Streamlit needing to serve it as a static
    file. Cached: reads the PNG from disk at most once per process,
    never touches chat data, and returns "" (rendered as a no-op) if the
    asset is ever missing so a missing file never crashes the app."""
    try:
        data = config.LOGO_PATH.read_bytes()
    except OSError:
        return ""
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _brand_logo_html(size: int = 28) -> str:
    """Return an `<img>` tag for the brand mark at the given pixel size,
    or "" if the logo asset isn't available (callers simply omit it)."""
    uri = _logo_data_uri()
    if not uri:
        return ""
    return f'<img class="st-brand-logo" src="{uri}" width="{size}" height="{size}" alt="" />'


def _main_header_brand() -> None:
    """Compact SHADOWTRACE wordmark + tagline shown once above the top
    navbar in the main content area, on every dashboard section —
    distinct from the sidebar's own brand lockup (`render_sidebar_brand`,
    only visible when the sidebar isn't collapsed/scrolled off on small
    screens) and from each page's own `_page_header(...)` title below
    it. Rendered exactly once per rerun, outside the section if/elif
    dispatch, so it never duplicates and never depends on which
    section is active. Pure markup — no state, no data access."""
    st.markdown(
        f"""
        <div class="st-main-header-brand">
            <div class="st-main-header-brand-word st-brand-row">{_brand_logo_html(28)}SHADOW<span>TRACE</span></div>
            <div class="st-main-header-brand-tag">Every conversation leaves a trace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _page_header(title: str, subtitle: str = "", hero: bool = False) -> None:
    """Lightweight page-level header: a title plus a one-line 'why this
    matters' description, per the WHAT-IS-THIS -> WHY-IT-MATTERS design
    hierarchy. `hero=True` (Overview only) renders the larger treatment;
    every other section gets the compact version so the brand doesn't
    repeat itself heavily on every rerun."""
    cls = "st-page-hero" if hero else "st-page-header"
    subtitle_html = f'<div class="st-page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="{cls}"><div class="st-page-title">{title}</div>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


@contextlib.contextmanager
def _loading_bar(message: str):
    """Show an animated horizontal loading bar in place of whatever
    would otherwise render here, for the duration of the wrapped
    block. Used around a section's data-computing calls (e.g.
    switching People sub-tabs) so the user sees an obvious "working on
    it" cue immediately, instead of the previous tab's stale cards
    lingering on screen mid-rerun with no feedback. The bar itself is
    indeterminate (an animated sweep, not a fake numeric percentage) —
    consistent with the rest of the app's "never fake progress" rule.
    Uses `st.empty()` so the placeholder is cleanly replaced by the
    real content the instant it's ready; the wrapped code's return
    value, caching, and computation are completely untouched."""
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="st-loading-bar-wrap">
            <div class="st-loading-bar-track"><div class="st-loading-bar-fill"></div></div>
            <div class="st-loading-bar-label">{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


def _panel_header(title: str, subtitle: str = "") -> None:
    """Small title (+ optional subtitle) rendered at the top of a
    `st.container(border=True)` panel, so a chart or block reads as a
    named card ("Conversation Pulse — Message volume over time")
    instead of a bare, unlabeled widget. Pure markup — no data access."""
    subtitle_html = f'<div class="st-panel-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="st-panel-title">{title}</div>{subtitle_html}',
        unsafe_allow_html=True,
    )


def _kpi_card(label: str, value, icon: str = "") -> None:
    """Premium KPI card matching the existing st.metric visual identity
    (see div[data-testid="stMetric"] in the theme CSS), with one added
    touch: whole-number values perform a pure-CSS count-up from 0 on
    render (the "@property --num" trick below — supported in the
    Chromium engine Streamlit's frontend runs on), instead of just
    appearing. This is presentation-only: `value` is always whatever
    the caller already computed from the real filtered data — nothing
    here recomputes, re-derives, or changes a single number. Non-integer
    values (already-formatted strings like "9 PM", or floats) fall back
    to a plain fade/slide reveal, per the spec's guidance to degrade
    gracefully when a true count-up isn't safely achievable.
    """
    safe_label = html.escape(str(label))
    icon_html = f'<span class="st-kpi-icon">{icon}</span> ' if icon else ""
    if isinstance(value, int) and not isinstance(value, bool):
        value_html = f'<div class="st-kpi-value st-kpi-count" style="--num:{value};"></div>'
    else:
        value_html = f'<div class="st-kpi-value st-kpi-static">{html.escape(str(value))}</div>'
    st.markdown(
        f"""
        <div class="st-kpi-card">
            <div class="st-kpi-label">{icon_html}{safe_label}</div>
            {value_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextlib.contextmanager
def _card(title: str = "", subtitle: str = ""):
    """Shared "chart card" wrapper used across every page: a native
    bordered container (see div[data-testid="stVerticalBlockBorderWrapper"]
    in the theme CSS) with an optional panel title/subtitle up top.
    Purely presentational — everything yielded inside still calls the
    exact same helper/analytics/charts functions as before; this only
    changes how the result is framed on the page."""
    with st.container(border=True):
        if title:
            _panel_header(title, subtitle)
        yield


def _chart(fig, **kwargs) -> None:
    """Render a Plotly figure inside its own card without repeating the
    card's title a second time. charts.py stamps every figure with its
    own `layout.title` (via `_apply_theme`) so it's still a sensible,
    fully-labelled chart if ever shown standalone -- but every call site
    in this file already wraps the chart in a `_card(...)`/`_panel_header(
    ...)` panel with that exact same title immediately above it, so the
    figure's built-in title only ever produced a visible duplicate here.
    This strips just the redundant on-canvas title (one line, presentation
    only) before handing the figure to `st.plotly_chart` -- the trace
    data, axis titles/labels, colors, and every computed value are
    untouched, and charts.py itself is not modified.

    Note: `title=""` (empty string), not `title=None` -- Plotly's
    frontend renders a `None`/null title as the literal text "undefined"
    instead of hiding it, which is worse than the duplicate we're fixing.
    """
    fig.update_layout(title="", margin=dict(l=10, r=10, t=10, b=10))
    kwargs.setdefault("width", "stretch")
    st.plotly_chart(fig, **kwargs)


def _initials(name: str) -> str:
    """Deterministic 1-2 letter avatar initials from a participant
    name (e.g. "Himanshu Kumar" -> "HK"). Presentation-only — never
    fabricates a photo or any data not already in the chat."""
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _render_avatar_cards(names: list[str], percents: list[float]) -> None:
    """Premium participant cards: avatar-initial circle, name, share of
    messages, and a progress bar — see People page. `percents` must
    already be each participant's real share of messages (0-100),
    computed by the caller from the actual filtered data."""
    for name, pct in zip(names, percents):
        pct = max(0.0, min(100.0, float(pct)))
        safe_name = html.escape(str(name))
        st.markdown(
            f"""
            <div class="st-avatar-card">
                <div class="st-avatar-circle">{_initials(str(name))}</div>
                <div class="st-avatar-body">
                    <div class="st-avatar-name">{safe_name}</div>
                    <div class="st-avatar-meta">{pct:.1f}% of messages</div>
                    <div class="st-avatar-bar"><div class="st-avatar-bar-fill" style="width:{pct}%"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _insight_list(items: list[str]) -> None:
    """Render a short list of already-computed, real insight strings as
    a compact card-style list (see 'Conversation Highlights'). Every
    string passed in must already be derived from the real filtered
    data by the caller — this function only lays them out."""
    marks = ["✓", "◉", "◈", "✦", "◆", "◷"]
    rows = "".join(
        f'<div class="st-insight-row"><span class="st-insight-mark">{marks[i % len(marks)]}</span>'
        f'<span>{text}</span></div>'
        for i, text in enumerate(items)
    )
    st.markdown(rows, unsafe_allow_html=True)


def _render_footer() -> None:
    st.markdown(
        f"""
        <div class="st-footer">
            <div class="st-footer-brand st-brand-row">{_brand_logo_html(18)}SHADOWTRACE</div>
            <div>Every conversation leaves a trace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
with nav.time_block("startup"):
    st.set_page_config(
        page_title="SHADOWTRACE",
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT,
    )
    utils.inject_custom_css()
    _apply_shadowtrace_theme()
    st.markdown('<div class="st-top-loading-bar"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UI stage state machine
# ---------------------------------------------------------------------------
# SHADOWTRACE has two major UI states, driven by one explicit,
# session_state-backed stage (never a local variable, so it survives
# every Streamlit rerun):
#   "landing"   - nothing uploaded yet: NO sidebar at all, just the
#                 hero + upload card.
#   "filtering" - a chat is uploaded and parsed: still no sidebar; the
#                 same landing page now also reveals "Filter your
#                 analysis" plus the Analyze Conversation button.
#   "analyzing" - the user clicked Analyze Conversation: a full-width
#                 loading screen is shown while the (already-cached)
#                 parsing step runs, then this flips itself to
#                 "dashboard".
#   "dashboard" - the real analytics workspace: sidebar (brand + nav
#                 rail + current-analysis/filters) + section content.
if "ui_stage" not in st.session_state:
    st.session_state["ui_stage"] = "landing"


def _reset_to_landing() -> None:
    """'Upload New Chat' handler. Clears every session_state flag tied
    to the previously analyzed chat — old filtered results, the AI
    chat transcript, and the stage itself — so nothing from the old
    chat can leak into the next upload. The underlying fingerprint-
    based cache invalidation in data_layer.py / rag.py already
    guarantees a *different* file never reuses a stale cache entry;
    this only resets the UI-facing session_state on top of that.
    """
    for key in (
        "ui_stage", "analyzed_once", "ai_chat_history",
        "chat_fingerprint", "raw_bytes", "raw_filename",
    ):
        st.session_state.pop(key, None)
    st.session_state["ui_stage"] = "landing"


def _render_shared_filters(options: dict) -> tuple:
    """The one and only advanced-filter UI. Rendered either inside the
    landing page's "Filter your analysis" card or the dashboard
    sidebar's "Filters" expander — never both in the same rerun, since
    those live in mutually exclusive ui_stage branches — so there is
    exactly one filtering implementation. Reusing the same widget keys
    in both places also means choices made before clicking Analyze
    Conversation carry straight through into the dashboard.
    """
    years = st.multiselect("Year", options["years"], key="f_years")
    months = st.multiselect("Month", options["months"], key="f_months")
    # "Week (ISO #)" intentionally not exposed in the UI — too
    # technical for normal users. The underlying filter still exists
    # in data_layer/utils, so it's always passed through as "no filter".
    weeks: list = []
    days = st.multiselect("Day of week", options["days"], key="f_days")
    date_range = st.date_input(
        "Date range",
        value=(options["min_date"], options["max_date"]),
        min_value=options["min_date"],
        max_value=options["max_date"],
        key="f_date_range",
    )
    hour_range = st.slider("Hour range", 0, 23, (0, 23), key="f_hour_range")
    return years, months, weeks, days, date_range, hour_range


# ===========================================================================
# STATE 1 — LANDING: upload + filter. NO sidebar is rendered here at
# all — not even the brand — per "there must be no sidebar on the
# landing page"; both branches below always end in st.stop() or
# st.rerun(), so the rest of the script (the dashboard) simply never
# executes while this stage is active.
# ===========================================================================
if st.session_state["ui_stage"] in ("landing", "filtering"):
    st.markdown(
        f"""
        <div class="st-landing">
          <div class="st-landing-bg">
            <div class="st-landing-grid"></div>
            <div class="st-landing-particles"></div>
          </div>
          <div class="st-landing-hero">
            <div class="st-landing-eyebrow">◆ Private conversation intelligence</div>
            <div class="st-landing-brand st-brand-row">{_brand_logo_html(48)}SHADOWTRACE</div>
            <div class="st-landing-tagline">Every conversation leaves a trace.</div>
            <div class="st-landing-desc">
                What was said is only the surface. Discover what lies beneath.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # st.container(key=...) is used (instead of a raw markdown div-open /
    # div-close pair) because it is the only way to get Streamlit to emit
    # a *real* wrapper element (carrying the "st-key-upload_card" class)
    # around the widgets below — plain st.markdown("<div>") / "</div>"
    # calls render as empty sibling nodes, not an actual parent, so none
    # of the descendant CSS selectors below would ever match.
    with st.container(key="upload_card"):
        st.markdown(
            """
            <div class="st-upload-icon-wrap">
              <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 16V4"></path>
                <path d="M6 10l6-6 6 6"></path>
                <path d="M4 20h16"></path>
              </svg>
            </div>
            <div class="st-upload-title">Upload your chat</div>
            <div class="st-upload-sub">Drop your WhatsApp .txt file here or click to browse</div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload your WhatsApp chat", type=["txt"],
            help="In WhatsApp: Chat > More options > Export chat > Without media",
            label_visibility="collapsed", key="landing_uploader",
        )
        st.markdown(
            '<div class="st-upload-hint">TXT · exported without media</div>',
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        _render_footer()
        st.stop()

    # Fingerprint the upload BEFORE decoding/parsing — this is the
    # single source of truth for "is this the same chat as before?"
    # that every cache (data_layer.py) and this session-state reset
    # key off.
    raw_bytes = uploaded_file.getvalue()
    fingerprint = data_layer.compute_fingerprint(raw_bytes)

    # A newly uploaded (different) chat must never show leftover UI
    # state from the previous one — a stale AI chat transcript that
    # quotes the old chat's messages, for instance. Comparing against
    # the fingerprint (not the filename) also means re-uploading the
    # SAME file is correctly treated as unchanged and keeps state.
    if st.session_state.get("chat_fingerprint") != fingerprint:
        st.session_state["chat_fingerprint"] = fingerprint
        st.session_state.pop("analyzed_once", None)
        st.session_state.pop("ai_chat_history", None)

    # Persisted past this block: once the stage moves on to
    # "analyzing" / "dashboard", the file_uploader widget above no
    # longer renders, so its value would otherwise disappear.
    st.session_state["raw_bytes"] = raw_bytes
    st.session_state["raw_filename"] = uploaded_file.name

    with nav.time_block("preprocessing"):
        raw_data, _decode_used_fallback = data_layer.decode_uploaded_chat(raw_bytes)
        df = data_layer.load_and_preprocess(raw_data, fingerprint)

    nav.set_perf_count("messages_processed", len(df))

    if df.empty or df['date'].isna().all():
        st.error(
            "We couldn't read any messages from this file. Make sure it's an "
            "unmodified WhatsApp chat export (12-hour clock format)."
        )
        _render_footer()
        st.stop()

    if _decode_used_fallback:
        st.warning(
            "This file wasn't standard UTF-8 text, so it was read with a "
            "fallback encoding — some characters may not display correctly."
        )

    st.session_state["ui_stage"] = "filtering"

    st.markdown(
        '<div class="st-ready-banner"><div class="st-ready-eyebrow">✓ Chat uploaded successfully</div></div>',
        unsafe_allow_html=True,
    )

    with st.container(key="filter_panel"):
        with st.container(border=True):
            _panel_header("Filter your analysis", "Narrow the conversation before analyzing — optional")

            user_list = df['user'].unique().tolist()
            if config.GROUP_NOTIFICATION_USER in user_list:
                user_list.remove(config.GROUP_NOTIFICATION_USER)
            user_list.sort()
            user_list.insert(0, "Overall")
            selected_user = st.selectbox("Participants", user_list, key="selected_user")

            options = utils.get_filter_options(df)
            f_years, f_months, f_weeks, f_days, f_date_range, f_hour_range = _render_shared_filters(options)

            active_filter_count = sum([
                bool(f_years), bool(f_months), bool(f_days),
                f_hour_range != (0, 23),
            ])
            if active_filter_count:
                st.caption(f"**{active_filter_count}** filter(s) active")

            b1, b2 = st.columns([1, 2])
            with b1:
                if st.button("Reset Filters", width='stretch'):
                    for k in ("f_years", "f_months", "f_days", "f_date_range", "f_hour_range", "selected_user"):
                        st.session_state.pop(k, None)
                    st.rerun()
            with b2:
                analyze_clicked = st.button("✦ Analyze Conversation", type="primary", width='stretch')

    st.caption(f"{df.shape[0]:,} total messages found in this file.")

    if analyze_clicked:
        st.session_state["ui_stage"] = "analyzing"
        st.rerun()

    _render_footer()
    st.stop()


# ===========================================================================
# STATE TRANSITION — ANALYZING: a full-width, indeterminate loading
# experience shown immediately after the click so the page never
# freezes silently. Copy is entirely human-facing; the animation below
# is CSS-only (see the .st-analyzing-* rules) so it costs nothing and
# never triggers a Streamlit rerun on its own.
# ===========================================================================
if st.session_state["ui_stage"] == "analyzing":
    st.markdown(
        f"""
        <div class="st-analyzing">
            <div class="st-analyzing-brand st-brand-row">{_brand_logo_html(36)}SHADOWTRACE</div>
            <div class="st-analyzing-title">Analyzing your conversation</div>
            <ul class="st-analyzing-list">
                <li>Reading messages</li>
                <li>Finding patterns</li>
                <li>Understanding activity</li>
                <li>Mapping participants</li>
                <li>Preparing insights</li>
            </ul>
            <div class="st-analyzing-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    raw_bytes = st.session_state.get("raw_bytes")
    fingerprint = st.session_state.get("chat_fingerprint")
    if raw_bytes is None or fingerprint is None:
        # Session state was cleared out from under us -- fail safe
        # back to the landing page instead of crashing on a missing key.
        _reset_to_landing()
        st.rerun()

    with nav.time_block("preprocessing"):
        raw_data, _ = data_layer.decode_uploaded_chat(raw_bytes)
        df = data_layer.load_and_preprocess(raw_data, fingerprint)

    # A short, one-time pause so the loading screen is actually
    # visible even when preprocessing itself is a near-instant cache
    # hit — this is not a fake progress percentage, just enough time
    # for the scanning/pulse animation above to register before the
    # page moves on to the dashboard.
    time.sleep(0.6)

    st.session_state["analyzed_once"] = True
    st.session_state["ui_stage"] = "dashboard"
    st.rerun()


# ===========================================================================
# STATE 2 — DASHBOARD: sidebar (brand + nav rail + current analysis +
# filters + Upload New Chat) plus whichever section is active.
# ===========================================================================
nav.render_sidebar_brand()

# Developer-only diagnostics. Never shown to normal users; flip
# config.PERFORMANCE_DEBUG on locally if you need to watch lazy
# execution / timings while developing.
# if getattr(config, "PERFORMANCE_DEBUG", False):
#     nav.render_debug_panel()

raw_bytes = st.session_state.get("raw_bytes")
fingerprint = st.session_state.get("chat_fingerprint")
if raw_bytes is None or fingerprint is None:
    # Shouldn't normally happen (only "analyzing"/"dashboard" reach
    # here, and both require raw_bytes+fingerprint), but fail safe
    # rather than crash if session_state was ever cleared externally.
    _reset_to_landing()
    st.rerun()

with nav.time_block("preprocessing"):
    raw_data, _ = data_layer.decode_uploaded_chat(raw_bytes)
    df = data_layer.load_and_preprocess(raw_data, fingerprint)

if df.empty or df['date'].isna().all():
    st.error("We couldn't read this conversation. Please upload it again.")
    if st.button("⬆ Upload New Chat"):
        _reset_to_landing()
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: current analysis (file summary + Upload New Chat)
# ---------------------------------------------------------------------------
user_list = df['user'].unique().tolist()
if config.GROUP_NOTIFICATION_USER in user_list:
    user_list.remove(config.GROUP_NOTIFICATION_USER)
user_list.sort()
user_list.insert(0, "Overall")

if nav.render_sidebar_analysis_summary(
    filename=st.session_state.get("raw_filename") or "your chat",
    message_count=len(df),
    participant_count=len(user_list) - 1,
):
    _reset_to_landing()
    st.rerun()

# ---------------------------------------------------------------------------
# Sidebar: user selection (existing feature, preserved)
# ---------------------------------------------------------------------------
selected_user = st.sidebar.selectbox("Participants", user_list, key="selected_user")

# ---------------------------------------------------------------------------
# Sidebar: advanced filters (same implementation as the landing page)
# ---------------------------------------------------------------------------
options = utils.get_filter_options(df)

with st.sidebar.expander("Filters", expanded=False):
    f_years, f_months, f_weeks, f_days, f_date_range, f_hour_range = _render_shared_filters(options)

    active_filter_count = sum([
        bool(f_years), bool(f_months), bool(f_days),
        f_hour_range != (0, 23),
    ])
    if active_filter_count:
        st.caption(f"**{active_filter_count}** filter(s) active")

    # Filtering is already reactive/cached (see get_filtered_df below),
    # so this button doesn't gate any computation — it's the explicit
    # "yes, use these filters" affordance from the reference design.
    st.button("Apply Filters", type="primary", width='stretch', key="apply_filters_btn")

st.sidebar.divider()
st.sidebar.caption(f"Total messages in file: **{df.shape[0]:,}**")


def get_filtered_df() -> pd.DataFrame:
    """Cached user + advanced-filter view of the chat.

    Delegates to `data_layer.get_filtered_view`, which is decorated
    with `@st.cache_data`. Every value that affects the result is
    passed in as a hashable, order-independent tuple, so:

    - Switching sections (no filter change) reuses the same cached
      DataFrame instead of re-running `.isin()` / boolean masks.
    - Changing exactly one filter only invalidates that specific
      (fingerprint, user, filters) cache entry — every other
      combination already computed this session stays cached.
    - A new upload (different fingerprint) can never reuse a filtered
      slice computed for the previous chat.
    """
    date_range = (
        tuple(f_date_range) if isinstance(f_date_range, tuple) and len(f_date_range) == 2 else None
    )
    return data_layer.get_filtered_view(
        df,
        fingerprint,
        selected_user,
        years=tuple(sorted(f_years)),
        months=tuple(sorted(f_months)),
        weeks=tuple(sorted(f_weeks)),
        days=tuple(sorted(f_days)),
        date_range=date_range,
        hour_range=tuple(f_hour_range),
    )


def get_ai_fingerprint() -> str:
    """Phase 6 — stable, O(1) cache key for the AI/RAG layer: the chat's
    own fingerprint + selected user + every active sidebar filter.

    This is deliberately NOT a hash of the filtered DataFrame's content
    (that's what rag.py falls back to if no fingerprint is passed in) —
    building it here costs nothing per rerun, and it changes if and only
    if the actual RAG-relevant selection changes:
        - a new file is uploaded            -> `fingerprint` changes
        - the sidebar user selector changes  -> `selected_user` changes
        - any advanced filter changes        -> the relevant f_* changes
    Switching AI sub-tabs, typing a new question, or running a different
    AI feature button does NOT change this string, so the vector index
    (see rag.get_or_build_index) is never rebuilt just because the user
    asked something new about the same selection.
    """
    date_range = (
        tuple(f_date_range) if isinstance(f_date_range, tuple) and len(f_date_range) == 2 else None
    )
    key_parts = (
        fingerprint, selected_user,
        tuple(sorted(f_years)), tuple(sorted(f_months)),
        tuple(sorted(f_weeks)), tuple(sorted(f_days)),
        date_range, tuple(f_hour_range),
    )
    return hashlib.md5(repr(key_parts).encode("utf-8")).hexdigest()


# =============================================================================
# SECTION RENDER FUNCTIONS
# =============================================================================
# Each function owns exactly one top-level nav section. app.py's main
# dispatch (bottom of this file) calls AT MOST ONE of these per rerun —
# that single call is the entire lazy-loading fix. Nothing here is
# wrapped in `st.tabs()`; every internal choice also uses `nav.sub_nav`
# so drilling into e.g. Insights → Sentiment does not also compute
# Topics/Toxicity/Word Similarity.


def render_overview(selected_user: str, filtered_df: pd.DataFrame) -> None:
    """📊 Overview — headline stats, KPIs, monthly/daily timelines."""
    nav.log_exec("render_overview")
    _page_header("Overview", "Your conversation at a glance.", hero=True)

    num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user, filtered_df)
    kpis = utils.compute_all_kpis(filtered_df)

    st.subheader("Headline Stats")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi_card("Total Messages", int(num_messages), icon="◆")
    with c2:
        _kpi_card("Total Words", int(num_words), icon="◉")
    with c3:
        _kpi_card("Media Shared", int(num_media), icon="◈")
    with c4:
        _kpi_card("Links Shared", int(num_links), icon="◷")

    st.subheader("Detailed KPIs")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        _kpi_card("Today's Messages", int(kpis["today_messages"]))
    with k2:
        _kpi_card("Avg Msgs / Day", kpis["avg_messages_per_day"])
    with k3:
        _kpi_card("Avg Words / Msg", kpis["avg_words_per_message"])
    with k4:
        _kpi_card("Longest Message", f"{kpis['longest_message']} words")
    with k5:
        _kpi_card("Most Active Hour", kpis["most_active_hour"])
    with k6:
        _kpi_card("Most Active Day", kpis["most_active_day"])

    st.divider()

    # Chart cards: each wrapped in a native bordered container (see
    # div[data-testid="stVerticalBlockBorderWrapper"] in the theme CSS)
    # so the chart reads as a single named panel — "Conversation Pulse"
    # — instead of a bare Plotly widget floating on the page. The
    # underlying data/calculations are untouched (still helper.py /
    # charts.py); only the wrapping presentation changed.
    pulse_col, side_col = st.columns([2, 1])
    with pulse_col:
        with st.container(border=True):
            _panel_header("Conversation Pulse", "Message volume over time")
            timeline = helper.monthly_timeline(selected_user, filtered_df)
            _chart(charts.monthly_timeline_chart(timeline), width='stretch')
    with side_col:
        with st.container(border=True):
            _panel_header("Conversation Highlights", "Generated from this selection")
            highlight_items = [
                f"Most active day is <b>{kpis['most_active_day']}</b>",
                f"Peak activity happens around <b>{kpis['most_active_hour']}</b>",
                f"Averaging <b>{kpis['avg_messages_per_day']}</b> messages per day",
                f"Longest message ran <b>{kpis['longest_message']}</b> words",
            ]
            _insight_list(highlight_items)

    with st.container(border=True):
        _panel_header("Daily Timeline", "Message volume per calendar date")
        daily = helper.daily_timeline(selected_user, filtered_df)
        _chart(charts.daily_timeline_chart(daily), width='stretch')


def render_activity(selected_user: str, filtered_df: pd.DataFrame, options: dict) -> None:
    """🗓️ Activity — daily/hourly patterns, weekend/peak, calendar, sleep.

    Sub-navigated: only the chosen sub-section's stats get computed.
    """
    nav.log_exec("render_activity")
    _page_header("Activity", "When do we talk? Explore daily and hourly patterns.")
    sub = nav.sub_nav(
        "Activity view",
        ["Daily & Hourly", "Weekend & Peak", "Sleep Schedule"],
        key="activity_sub",
    )

    if sub == "Daily & Hourly":
        nav.log_exec("render_activity:daily_hourly")
        col1, col2 = st.columns(2)
        with col1, _card("Most Busy Day"):
            busy_day = helper.week_activity_map(selected_user, filtered_df)
            _chart(charts.week_activity_chart(busy_day), width='stretch')
        with col2, _card("Most Busy Month"):
            busy_month = helper.month_activity_map(selected_user, filtered_df)
            _chart(charts.month_activity_chart(busy_month), width='stretch')

        with _card("Hourly Analysis", "Messages by hour of day"):
            hourly = helper.hourly_activity(selected_user, filtered_df)
            _chart(charts.hourly_activity_chart(hourly), width='stretch')

    elif sub == "Weekend & Peak":
        nav.log_exec("render_activity:weekend_peak")
        with _card("Weekend vs Weekday"):
            weekend_df = analytics.weekend_vs_weekday(selected_user, filtered_df)
            if weekend_df.empty:
                st.info("No data for this selection.")
            else:
                _chart(charts.weekend_weekday_chart(weekend_df), width='stretch')
                st.dataframe(weekend_df, width='stretch')

        with _card("Peak Activity", "Busiest and quietest hours in this selection"):
            hourly_full = analytics.hourly_activity_full(filtered_df)
            top_hours = analytics.peak_hours(selected_user, filtered_df, top_n=5)
            least_hours = analytics.quiet_hours(selected_user, filtered_df, top_n=5)

            p1, p2 = st.columns(2)
            p1.metric("◆ Top Hour", f"{top_hours.index[0]:02d}:00" if not top_hours.empty else "N/A")
            p2.metric("◷ Quietest Hour", f"{least_hours.index[0]:02d}:00" if not least_hours.empty else "N/A")
            _chart(charts.peak_activity_chart(hourly_full, top_hours, least_hours), width='stretch')
            st.caption("Highlighted bars: green = top 5 busiest hours, grey = 5 quietest hours.")

    elif sub == "Sleep Schedule":
        nav.log_exec("render_activity:sleep")
        with _card("Estimated Sleep Schedule", "A heuristic read on the quietest hours"):
            sleep_info = analytics.estimate_sleep_schedule(selected_user, filtered_df)
            if sleep_info["sleep_start"] is None:
                st.info("Not enough data to estimate a sleep schedule.")
            else:
                s1, s2 = st.columns(2)
                s1.metric("◆ Estimated Sleep Time", f"{sleep_info['sleep_start']:02d}:00")
                s2.metric("◉ Estimated Wake Time", f"{sleep_info['wake_up']:02d}:00")
                hourly_full_sleep = analytics.hourly_activity_full(filtered_df)
                _chart(
                    charts.sleep_schedule_chart(hourly_full_sleep, sleep_info['sleep_start'], sleep_info['wake_up']),
                    width='stretch',
                )
                st.caption(
                    "Heuristic estimate: the quietest "
                    f"{sleep_info['window_hours']}-hour window in the messaging pattern, not a real sleep measurement."
                )


def render_users(selected_user: str, filtered_df: pd.DataFrame, options: dict) -> None:
    """👥 People — busiest users, head-to-head comparison, engagement score."""
    nav.log_exec("render_users")
    _page_header("People", "Who drives this conversation?")
    sub = nav.sub_nav(
        "People view", ["Most Busy Users", "Compare Users", "Engagement Score"], key="people_sub",
    )

    if sub == "Most Busy Users":
        nav.log_exec("render_users:busy_users")
        if selected_user == 'Overall':
            with _card("Most Busy Users", "Ranked by message count"):
                with _loading_bar("Finding your most active people..."):
                    x, new_df = helper.most_busy_users(filtered_df)
                top_ranked = new_df.sort_values("percent", ascending=False).head(6)
                _render_avatar_cards(top_ranked["name"].tolist(), top_ranked["percent"].tolist())
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    _chart(charts.most_busy_users_chart(x), width='stretch')
                with col2:
                    st.dataframe(new_df, width='stretch', height=400)
        else:
            st.info("Select **Overall** in the sidebar to see per-user activity comparisons.")

    elif sub == "Compare Users":
        nav.log_exec("render_users:compare")
        with _card("Compare Two Users"):
            compare_users_list = options["users"]
            if len(compare_users_list) < 2:
                st.info("Need at least two participants in the chat to compare.")
            else:
                c1, c2 = st.columns(2)
                user_a = c1.selectbox("User A", compare_users_list, index=0, key="cmp_user_a")
                default_b = 1 if len(compare_users_list) > 1 else 0
                user_b = c2.selectbox("User B", compare_users_list, index=default_b, key="cmp_user_b")

                if user_a == user_b:
                    st.warning("Pick two different users to compare.")
                else:
                    with _loading_bar("Comparing these two people..."):
                        comparison_df = analytics.compare_users(filtered_df, user_a, user_b)
                    _chart(charts.user_comparison_chart(comparison_df), width='stretch')
                    st.dataframe(comparison_df, width='stretch')

    elif sub == "Engagement Score":
        nav.log_exec("render_users:engagement")
        with _card("Engagement Score", "0-100 ranking across message volume, words, media, and consistency"):
            if selected_user != "Overall":
                st.info("Select **Overall** in the sidebar to see engagement scores across all users.")
            else:
                with _loading_bar("Scoring everyone's engagement..."):
                    scores_df = analytics.engagement_scores(filtered_df)
                if scores_df.empty:
                    st.info("Not enough data to compute engagement scores.")
                else:
                    _chart(charts.engagement_score_chart(scores_df), width='stretch')
                    st.dataframe(scores_df, width='stretch')
                    st.caption(
                        "A relative 0-100 ranking blending message volume, words, media, emoji, links, "
                        "response speed, and day-to-day consistency."
                    )


def render_search(filtered_df: pd.DataFrame, options: dict) -> None:
    """🔍 Search — keyword / regex / user / date search over messages.

    Its own function (as requested) so it can be called both from the
    Conversations section's sub-nav and, if ever needed, standalone —
    without dragging in every other Conversations feature.
    """
    nav.log_exec("render_search")
    with _card("Search Messages", "Keyword, regex, user, and date filters"):
        s1, s2 = st.columns([3, 1])
        with s1:
            keyword = st.text_input("Keyword or phrase", placeholder="e.g. good morning")
        with s2:
            use_regex = st.checkbox("Regex mode")

        search_user = st.selectbox("Filter by user", ["Overall"] + options["users"], key="search_user")
        search_date = st.date_input(
            "Filter by date (optional)", value=None,
            min_value=options["min_date"], max_value=options["max_date"],
            key="search_date",
        )

        try:
            search_results = utils.search_messages(
                filtered_df,
                keyword=keyword,
                user=search_user,
                date=search_date,
                use_regex=use_regex,
            )
            st.caption(f"{len(search_results):,} matching message(s)")
            st.dataframe(search_results, width='stretch', height=420)
        except re.error as exc:
            st.error(f"Invalid regex pattern: {exc}")


def render_content(selected_user: str, filtered_df: pd.DataFrame, options: dict) -> None:
    """💬 Conversations — word cloud, emoji, starters, phrases, response
    time, weekly report, and message search — all sub-navigated so only
    the chosen sub-feature actually computes."""
    nav.log_exec("render_content")
    _page_header("Conversations", "How do we communicate?")
    sub = nav.sub_nav(
        "Conversations view",
        [
            "Word Cloud & Common Words", "Emoji", "Emoji Timeline", "Starters & Frequency",
            "Message Stats", "Phrases & Domains", "Response Time", "Weekly Report", "Search",
        ],
        key="conv_sub",
    )

    if sub == "Word Cloud & Common Words":
        nav.log_exec("render_content:wordcloud")
        with _card("Word Cloud"):
            wc_image = helper.create_wordcloud(selected_user, filtered_df)
            st.image(wc_image, width='stretch')

        with _card("Most Common Words"):
            most_common_df = helper.most_common_words(selected_user, filtered_df)
            if most_common_df.empty:
                st.info("Not enough text data to compute common words for this selection.")
            else:
                _chart(charts.most_common_words_chart(most_common_df), width='stretch')

    elif sub == "Emoji":
        nav.log_exec("render_content:emoji")
        with _card("Emoji Analysis"):
            emoji_df = helper.emoji_helper(selected_user, filtered_df)
            if emoji_df.empty:
                st.info("No emoji found for this selection.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(emoji_df, width='stretch', height=400)
                with col2:
                    _chart(charts.emoji_pie_chart(emoji_df), width='stretch')

    elif sub == "Emoji Timeline":
        nav.log_exec("render_content:emoji_timeline")
        with _card("Emoji Usage Over Time"):
            monthly_emoji = analytics.emoji_timeline_monthly(selected_user, filtered_df)
            if monthly_emoji.empty:
                st.info("No emoji found for this selection.")
            else:
                _chart(charts.emoji_timeline_chart(monthly_emoji), width='stretch')

        if selected_user == "Overall":
            with _card("Emoji Usage by User"):
                by_user_timeline = analytics.emoji_timeline_by_user(filtered_df)
                if by_user_timeline.empty:
                    st.info("No emoji found for this selection.")
                else:
                    _chart(charts.emoji_timeline_by_user_chart(by_user_timeline), width='stretch')
        else:
            st.caption("Select **Overall** in the sidebar to compare emoji usage across users.")

    elif sub == "Starters & Frequency":
        nav.log_exec("render_content:starters")
        with _card("Conversation Starters"):
            starters = analytics.conversation_starters(filtered_df)
            if starters.empty:
                st.info("No distinct conversations detected in this selection.")
            else:
                _chart(charts.conversation_starters_chart(starters), width='stretch')

        with _card("Starters Over Time"):
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
            starter_freq_choice = st.radio(
                "Group starters by", list(freq_map.keys()), horizontal=True, key="starter_freq"
            )
            starters_trend = analytics.conversation_starters_by_period(filtered_df, freq=freq_map[starter_freq_choice])
            if starters_trend.empty:
                st.info("Not enough data to chart starters over time.")
            else:
                _chart(charts.conversation_starters_trend_chart(starters_trend), width='stretch')

        with _card("Message Frequency"):
            msg_freq_choice = st.radio("Group messages by", list(freq_map.keys()), horizontal=True, key="msg_freq")
            msg_freq_df = analytics.message_frequency(selected_user, filtered_df, freq=freq_map[msg_freq_choice])
            if msg_freq_df.empty:
                st.info("No messages to chart.")
            else:
                _chart(charts.message_frequency_chart(msg_freq_df, msg_freq_choice), width='stretch')

    elif sub == "Message Stats":
        nav.log_exec("render_content:msgstats")
        with _card("Average Message Length"):
            length_df = analytics.message_length_stats(selected_user, filtered_df)
            if length_df.empty:
                st.info("Not enough text messages in this selection.")
            else:
                _chart(charts.message_length_chart(length_df), width='stretch')
                st.dataframe(length_df, width='stretch')

        with _card("Longest Message"):
            longest = analytics.longest_message_details(selected_user, filtered_df)
            if longest["sender"] == "N/A":
                st.info("No text messages found in this selection.")
            else:
                l1, l2, l3 = st.columns(3)
                l1.metric("◆ Sender", longest["sender"])
                l2.metric("◉ Word Count", longest["word_count"])
                l3.metric("◈ Character Count", longest["char_count"])
                st.caption(f"Sent on {longest['date']}")
                st.text_area("Message", longest["message"], height=150, disabled=True)

    elif sub == "Phrases & Domains":
        nav.log_exec("render_content:phrases")
        with _card("Most Used Phrases"):
            phrase_type = st.radio("Phrase length", ["Bigrams", "Trigrams"], horizontal=True)
            n = 2 if phrase_type == "Bigrams" else 3
            ngram_df = analytics.top_ngrams(selected_user, filtered_df, n=n, top_n=config.NGRAM_TOP_N)
            if ngram_df.empty:
                st.info("Not enough text to compute phrases for this selection.")
            else:
                _chart(charts.ngram_chart(ngram_df), width='stretch')

        with _card("Shared Domains"):
            domains_df = analytics.shared_domains(selected_user, filtered_df, top_n=15)
            if domains_df.empty:
                st.info("No links found for this selection.")
            else:
                _chart(charts.domains_chart(domains_df), width='stretch')

    elif sub == "Response Time":
        nav.log_exec("render_content:response_time")
        resp_stats = analytics.response_time_stats(selected_user, filtered_df)

        if resp_stats["reply_count"] == 0:
            with _card("Response Time Analysis"):
                st.info("Not enough back-and-forth replies in this selection to compute response times.")
        else:
            with _card("Response Time Analysis"):
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("◆ Avg Reply Time", f"{resp_stats['avg_minutes']} min")
                r2.metric("◉ Median Reply Time", f"{resp_stats['median_minutes']} min")
                r3.metric("◈ Fastest Responder", resp_stats['fastest_user'], f"{resp_stats['fastest_minutes']} min")
                r4.metric("◷ Slowest Responder", resp_stats['slowest_user'], f"{resp_stats['slowest_minutes']} min")

            col1, col2 = st.columns(2)
            with col1, _card("Avg Reply Time by User"):
                by_user = analytics.response_time_by_user(filtered_df)
                if by_user.empty:
                    st.info("No per-user reply data available.")
                else:
                    _chart(
                        charts.ranking_bar_chart(by_user, "Avg Reply Time by User", "Minutes"),
                        width='stretch',
                    )
            with col2, _card("Reply Time Distribution"):
                hist_data = analytics.response_time_histogram_data(selected_user, filtered_df)
                if hist_data.empty:
                    st.info("No reply data available.")
                else:
                    _chart(charts.response_time_histogram_chart(hist_data), width='stretch')

            with _card("Response Time Trend"):
                trend = analytics.response_time_trend(selected_user, filtered_df)
                if trend.empty:
                    st.info("Not enough data to chart a trend.")
                else:
                    _chart(charts.response_time_trend_chart(trend), width='stretch')

                st.caption(
                    f"Based on {resp_stats['reply_count']:,} timed replies "
                    f"(gaps over {config.REPLY_GAP_CAP_MINUTES} minutes count as a new conversation, not a reply)."
                )

    elif sub == "Weekly Report":
        nav.log_exec("render_content:weekly_report")
        with _card("Weekly Report"):
            report_text = analytics.generate_weekly_report(selected_user, filtered_df)
            st.markdown(report_text)

    elif sub == "Search":
        render_search(filtered_df, options)


def render_nlp(selected_user: str, filtered_df: pd.DataFrame) -> None:
    """🧠 Insights — Phase 4 NLP features: sentiment, emotion/intent,
    topics, keywords, summarization, toxicity, language/readability/spam,
    word similarity. Sub-navigated so, e.g., opening Sentiment never
    triggers Topics/Toxicity/Word Similarity."""
    nav.log_exec("render_nlp")
    _page_header("Insights", "The analytical workspace — sentiment, topics, tone, and more.")
    sub = nav.sub_nav(
        "Insight",
        [
            "Sentiment", "Emotion & Intent", "Topics", "Keywords", "Summarizer",
            "Toxicity", "Language & Quality", "Word Similarity",
        ],
        key="insights_sub",
    )

    if sub == "Sentiment":
        nav.log_exec("render_nlp:sentiment")
        with _card("Sentiment Distribution", "Share of positive, neutral, and negative messages"):
            with st.spinner("Understanding conversation tone..."):
                dist = sentiment.sentiment_distribution(selected_user, filtered_df)
            if dist.empty or dist['count'].sum() == 0:
                st.info("Not enough text messages to analyze sentiment.")
            else:
                s1, s2, s3 = st.columns(3)
                s1.metric("😊 Positive", int(dist.loc[dist['sentiment'] == 'Positive', 'count'].sum()))
                s2.metric("😐 Neutral", int(dist.loc[dist['sentiment'] == 'Neutral', 'count'].sum()))
                s3.metric("😠 Negative", int(dist.loc[dist['sentiment'] == 'Negative', 'count'].sum()))
                _chart(
                    charts.distribution_pie_chart(dist, 'sentiment', 'count', "Sentiment Distribution"),
                    width='stretch',
                )

        with _card("Sentiment Over Time"):
            sent_freq_choice = st.radio("Group by", ["Daily", "Weekly", "Monthly"], horizontal=True, key="sent_freq")
            sent_freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
            with st.spinner("Understanding conversation tone..."):
                sent_timeline = sentiment.sentiment_timeline(
                    selected_user, filtered_df, freq=sent_freq_map[sent_freq_choice],
                )
            if sent_timeline.empty:
                st.info("Not enough data to chart a sentiment trend.")
            else:
                _chart(charts.sentiment_timeline_chart(sent_timeline), width='stretch')

        if selected_user == "Overall":
            with _card("Sentiment by User"):
                by_user_sent = sentiment.sentiment_by_user(filtered_df)
                if by_user_sent.empty:
                    st.info("Not enough data to compare users.")
                else:
                    st.dataframe(by_user_sent, width='stretch')

        st.caption("Each message is scored from -1 (very negative) to +1 (very positive).")

    elif sub == "Emotion & Intent":
        nav.log_exec("render_nlp:emotion_intent")
        col1, col2 = st.columns(2)
        with col1, _card("Emotion Detection"):
            with st.spinner("Understanding conversation tone..."):
                emo_dist = sentiment.emotion_distribution(selected_user, filtered_df)
            if emo_dist.empty:
                st.info("Not enough text to detect emotions.")
            else:
                _chart(
                    charts.distribution_pie_chart(emo_dist, 'emotion', 'count', "Emotion Distribution"),
                    width='stretch',
                )
        with col2, _card("Intent Classification"):
            with st.spinner("Understanding conversation tone..."):
                intent_dist = topics.intent_distribution(selected_user, filtered_df)
            if intent_dist.empty:
                st.info("Not enough text to classify intent.")
            else:
                _chart(
                    charts.distribution_pie_chart(intent_dist, 'intent', 'count', "Intent Distribution"),
                    width='stretch',
                )

        if selected_user == "Overall":
            with _card("Emotion by User"):
                emo_by_user_df = sentiment.emotion_by_user(filtered_df)
                if not emo_by_user_df.empty:
                    _chart(
                        charts.grouped_stack_chart(emo_by_user_df, "Emotion by User", "Messages"),
                        width='stretch',
                    )

            with _card("Intent by User"):
                intent_by_user_df = topics.intent_by_user(filtered_df)
                if not intent_by_user_df.empty:
                    _chart(
                        charts.grouped_stack_chart(intent_by_user_df, "Intent by User", "Messages"),
                        width='stretch',
                    )

        st.caption("Emotion: Happy / Sad / Angry / Excited. Intent: Study / Work / Travel / Shopping / Movies / Sports.")

    elif sub == "Topics":
        nav.log_exec("render_nlp:topics")
        with _card("Discussion Topics", "Themes grouped from what's actually discussed most"):
            n_topics_choice = st.slider("Number of topics", 2, 10, 5)
            with st.spinner("Discovering conversation themes..."):
                topics_df = topics.topic_model(selected_user, filtered_df, n_topics=n_topics_choice)
            if topics_df.empty:
                st.info(
                    "Not enough distinct messages in this selection for topic modeling "
                    "(try widening the filters or picking Overall)."
                )
            else:
                _chart(charts.topic_size_chart(topics_df), width='stretch')
                st.dataframe(topics_df, width='stretch')
            st.caption("Groups messages into themes based on what's actually discussed most.")

    elif sub == "Keywords":
        nav.log_exec("render_nlp:keywords")
        with _card("Keyword Extraction"):
            kw_method = st.radio("Method", ["TF-IDF", "YAKE", "KeyBERT"], horizontal=True)
            top_n_kw = st.slider("Number of keywords", 5, 40, 20, key="kw_topn")

            if kw_method == "TF-IDF":
                with st.spinner("Discovering conversation themes..."):
                    kw_df = topics.keywords_tfidf(selected_user, filtered_df, top_n=top_n_kw)
            elif kw_method == "YAKE":
                with st.spinner("Discovering conversation themes..."):
                    kw_df = topics.keywords_yake(selected_user, filtered_df, top_n=top_n_kw)
                if not topics.YAKE_AVAILABLE:
                    st.caption("Showing results from the standard method instead.")
            else:
                with st.spinner("Discovering conversation themes..."):
                    kw_df = topics.keywords_keybert(selected_user, filtered_df, top_n=top_n_kw)
                if not topics.KEYBERT_AVAILABLE:
                    st.caption("Showing results from the standard method instead.")

            if kw_df.empty:
                st.info("Not enough text to extract keywords.")
            else:
                _chart(charts.keyword_bar_chart(kw_df, f"Top Keywords ({kw_method})"), width='stretch')

        with _card("Most Important Words"):
            important_scope = st.radio("Breakdown", ["By User", "By Month"], horizontal=True)
            if important_scope == "By User":
                important_df = sentiment.important_words_by_user(filtered_df)
            else:
                important_df = sentiment.important_words_by_month(selected_user, filtered_df)
            if important_df.empty:
                st.info("Not enough grouped text data for this breakdown (need at least 2 groups).")
            else:
                st.dataframe(important_df, width='stretch', height=400)
            st.caption(
                "TF-IDF weighted, so a word distinctive to one user/month ranks above words everyone uses equally."
            )

    elif sub == "Summarizer":
        nav.log_exec("render_nlp:summarizer")
        with _card("Text Summarization"):
            summary_mode = st.radio(
                "Summarize", ["Selected Chat (current filters)", "Last 7 Days", "Last 30 Days"],
            )
            n_sentences = st.slider("Summary length (sentences)", 3, 12, 6)

            if summary_mode == "Selected Chat (current filters)":
                with st.spinner("Preparing your conversation summary..."):
                    summary_text = topics.summarize_chat(selected_user, filtered_df, n_sentences=n_sentences)
            elif summary_mode == "Last 7 Days":
                with st.spinner("Preparing your conversation summary..."):
                    summary_text = topics.summarize_period(
                        selected_user, filtered_df, period="W", n_sentences=n_sentences,
                    )
            else:
                with st.spinner("Preparing your conversation summary..."):
                    summary_text = topics.summarize_period(
                        selected_user, filtered_df, period="M", n_sentences=n_sentences,
                    )

            st.markdown(f"> {summary_text}")
            st.caption(
                "Extractive summarization: ranks sentences by TF-IDF weight and keeps the top ones in their "
                "original order — no external model or API call required."
            )

    elif sub == "Toxicity":
        nav.log_exec("render_nlp:toxicity")
        with _card("Toxicity Overview"):
            with st.spinner("Reviewing conversation safety..."):
                tox_summary_df = toxicity.toxicity_summary(selected_user, filtered_df)
            if tox_summary_df.empty:
                st.info("No text messages to analyze.")
            else:
                _chart(
                    charts.distribution_pie_chart(tox_summary_df, 'label', 'count', "Toxicity Labels"),
                    width='stretch',
                )

        with _card("Flagged Messages"):
            with st.spinner("Reviewing conversation safety..."):
                flagged_df = toxicity.toxicity_table(selected_user, filtered_df, only_flagged=True)
            if flagged_df.empty:
                st.success("No abusive, offensive, or hateful messages detected in this selection. 🎉")
            else:
                st.dataframe(flagged_df, width='stretch', height=350)

        if selected_user == "Overall":
            with _card("Flagged Rate by User"):
                tox_by_user_df = toxicity.toxicity_by_user(filtered_df)
                if not tox_by_user_df.empty:
                    st.dataframe(tox_by_user_df, width='stretch')

        st.caption("Flags messages that may be abusive, offensive, or hateful.")

    elif sub == "Language & Quality":
        nav.log_exec("render_nlp:language_quality")
        with _card("Language Detection"):
            with st.spinner("Reading the messaging style..."):
                lang_dist = nlp_helper.language_distribution(selected_user, filtered_df)
            if lang_dist.empty:
                st.info("Not enough text to detect language.")
            else:
                lang_df = lang_dist.reset_index()
                lang_df.columns = ['language', 'count']
                _chart(
                    charts.distribution_pie_chart(lang_df, 'language', 'count', "Language Mix"),
                    width='stretch',
                )

        with _card("Readability", "Flesch Reading Ease: 90-100 very easy, 60-70 plain English, 0-30 very difficult"):
            if selected_user == "Overall":
                read_df = nlp_helper.readability_by_user(filtered_df)
                if read_df.empty:
                    st.info("Not enough text to score readability.")
                else:
                    st.dataframe(read_df, width='stretch')
            else:
                real_msgs = nlp_helper.clean_messages(filtered_df, selected_user)
                if real_msgs.empty:
                    st.info("Not enough text to score readability.")
                else:
                    avg_score = real_msgs['message'].apply(nlp_helper.readability_score).mean()
                    st.metric("◆ Avg Flesch Reading Ease", round(avg_score, 2))

        with _card("Spam Detection"):
            with st.spinner("Reviewing conversation safety..."):
                spam_info = toxicity.spam_summary(selected_user, filtered_df)
            sp1, sp2, sp3 = st.columns(3)
            sp1.metric("◆ Total Messages", spam_info["total_messages"])
            sp2.metric("◈ Flagged as Spam", spam_info["flagged_messages"])
            sp3.metric("◷ Spam Rate", f"{spam_info['spam_rate_pct']}%")
            with st.spinner("Reviewing conversation safety..."):
                spam_df = toxicity.spam_table(selected_user, filtered_df)
            if spam_df.empty:
                st.success("No spam-like messages detected in this selection.")
            else:
                st.dataframe(spam_df, width='stretch', height=300)
            st.caption(
                "Heuristic signals: link floods, repeated characters, ALL-CAPS shouting, promotional keywords, "
                "and near-duplicate messages sent 3+ times."
            )

    elif sub == "Word Similarity":
        nav.log_exec("render_nlp:word_similarity")
        with _card(
            "Word Similarity",
            "Distributional similarity from this chat's own context vectors",
        ):
            sim1, sim2 = st.columns(2)
            with sim1:
                word_a = st.text_input("Word A", value="")
            with sim2:
                word_b = st.text_input("Word B (optional — leave blank to rank similar words instead)", value="")

            if word_a and word_b:
                with st.spinner("Preparing your conversation for search..."):
                    sim_score = nlp_helper.word_similarity(filtered_df, selected_user, word_a, word_b)
                if sim_score is None:
                    st.warning("One or both words weren't found in this selection's messages.")
                else:
                    st.metric(f"Similarity: '{word_a}' ↔ '{word_b}'", sim_score)
            elif word_a:
                with st.spinner("Preparing your conversation for search..."):
                    similar_df = nlp_helper.most_similar_words(filtered_df, selected_user, word_a, top_n=15)
                if similar_df.empty:
                    st.warning(f"'{word_a}' wasn't found in this selection's messages (needs at least 2 occurrences).")
                else:
                    st.dataframe(similar_df, width='stretch')
            else:
                st.info("Enter a word above to see its most similar words in this chat.")


def render_ai(selected_user: str, filtered_df: pd.DataFrame, options: dict, ai_fingerprint: str) -> None:
    """🤖 AI Assistant — Phase 5 UI, Phase 6 on-demand loading.

    Opening this tab builds NOTHING: no LLM client, no embeddings model,
    no RAG/vector index. The availability banner below uses
    `ai_helper.is_ai_configured()` (a plain key-presence check) instead
    of `is_ai_available()`, specifically because the latter would
    instantiate a real LangChain provider client just to render a
    caption. Every sub-feature below only touches the LLM/RAG stack
    inside its own button/submit handler — see ai_helper.py / rag.py /
    vector_store.py, which cache the LLM client, the embeddings model,
    and the vector index independently so none of them is rebuilt
    because a *different* one of those is used.
    """
    nav.log_exec("render_ai")
    _page_header("AI Assistant", "Ask anything about your conversation.")

    if not ai_helper.is_ai_configured():
        st.warning(ai_helper.AI_SETUP_MESSAGE)

    sub = nav.sub_nav(
        "AI feature",
        [
            "Summaries", "Ask & Chat", "Insights & Recommendations",
            "Semantic Search", "Highlights", "People Analysis",
        ],
        key="ai_sub",
    )

    if sub == "Summaries":
        nav.log_exec("render_ai:summaries")
        with _card("Chat Summary", "Daily, weekly, monthly, or the whole selection"):
            summary_choice = st.radio(
                "Summary type", ["Daily", "Weekly", "Monthly", "Whole Selection (Chat Summary)"],
                horizontal=True, key="ai_summary_choice",
            )
            if st.button("Generate Summary", key="ai_summary_btn", type="primary"):
                with st.spinner("Thinking through the conversation..."):
                    if summary_choice == "Daily":
                        result = ai_helper.ai_daily_summary(selected_user, filtered_df)
                    elif summary_choice == "Weekly":
                        result = ai_helper.ai_weekly_summary(selected_user, filtered_df)
                    elif summary_choice == "Monthly":
                        result = ai_helper.ai_monthly_summary(selected_user, filtered_df)
                    else:
                        result = ai_helper.ai_chat_summary(selected_user, filtered_df)
                st.markdown(result)

    elif sub == "Ask & Chat":
        nav.log_exec("render_ai:ask_chat")

        with _card("Ask About Your Conversation"):
            if "ai_chat_history" not in st.session_state:
                st.session_state["ai_chat_history"] = []

            if not st.session_state["ai_chat_history"]:
                st.caption("Ask anything about your conversation — try one of the suggestions below.")

            for q, a in st.session_state["ai_chat_history"]:
                st.chat_message("user").markdown(q)
                st.chat_message("assistant").markdown(a)

            suggested_questions = [
                "What were the main topics?",
                "Who was most active?",
                "When was the conversation most positive?",
                "What patterns stand out?",
            ]
            st.caption("Suggested questions")
            picked_question = None
            with st.container(key="chip_row"):
                chip_cols = st.columns(len(suggested_questions))
                for col, q in zip(chip_cols, suggested_questions):
                    if col.button(q, key=f"ai_chip_{q}"):
                        picked_question = q

            if st.session_state["ai_chat_history"] and st.button("Clear conversation", key="ai_clear_history"):
                st.session_state["ai_chat_history"] = []
                st.rerun()

        typed_question = st.chat_input("Ask about your conversation...")
        user_question = picked_question or typed_question
        if user_question:
            st.chat_message("user").markdown(user_question)
            with st.spinner("Thinking about your conversation..."):
                answer = ai_helper.ask_question(
                    user_question, selected_user, filtered_df,
                    chat_history=st.session_state["ai_chat_history"],
                    fingerprint=ai_fingerprint,
                )
            st.chat_message("assistant").markdown(answer)
            st.session_state["ai_chat_history"].append((user_question, answer))
            st.rerun()

    elif sub == "Insights & Recommendations":
        nav.log_exec("render_ai:insights_recommendations")
        col1, col2 = st.columns(2)
        with col1, _card("AI Insights", "Non-obvious patterns across your metrics"):
            if st.button("Generate Insights", key="ai_insights_btn", width='stretch'):
                with st.spinner("Analyzing patterns..."):
                    st.markdown(ai_helper.generate_insights(selected_user, filtered_df))
        with col2, _card("AI Recommendations", "Five concrete, data-backed suggestions"):
            if st.button("Generate Recommendations", key="ai_reco_btn", width='stretch'):
                with st.spinner("Thinking of suggestions..."):
                    st.markdown(ai_helper.ai_recommendations(selected_user, filtered_df))

        with _card("Conversation Quality Score"):
            if st.button("Compute Quality Score", key="ai_quality_btn"):
                with st.spinner("Scoring the conversation..."):
                    quality = ai_helper.conversation_quality_score(selected_user, filtered_df)
                st.metric("◆ Quality Score", f"{quality['score']} / 100")
                st.markdown(quality["narrative"])
                with st.expander("Signal breakdown"):
                    st.json(quality["signals"])

    elif sub == "Semantic Search":
        nav.log_exec("render_ai:semantic_search")
        with _card("Search Your Conversation", "Finds moments by meaning, not just exact wording"):
            search_query = st.text_input("Search query", key="ai_semantic_query")
            top_k = st.slider("Number of results", 3, 20, 8, key="ai_semantic_k")
            if search_query:
                with st.spinner("Preparing your conversation for search..."):
                    hits = ai_helper.semantic_search(filtered_df, search_query, k=top_k, fingerprint=ai_fingerprint)
                if not hits:
                    st.info("No relevant messages found — try a different query or widen the filters.")
                else:
                    for i, hit in enumerate(hits, 1):
                        score_label = f" (relevance: {hit['score']})" if hit["score"] is not None else ""
                        st.markdown(f"**{i}.**{score_label}")
                        meta = hit.get("metadata") or {}
                        if meta.get("users") and meta.get("start_date"):
                            span = (
                                meta["start_date"] if meta["start_date"] == meta.get("end_date")
                                else f"{meta['start_date']} to {meta.get('end_date')}"
                            )
                            st.caption(f"{', '.join(meta['users'])} · {span}")
                        st.text(hit["text"])

    elif sub == "Highlights":
        nav.log_exec("render_ai:highlights")
        with _card("Smart Conversation Highlights", "Funny exchanges, decisions, and turning points"):
            if st.button("Find Highlights", key="ai_highlights_btn", type="primary"):
                with st.spinner("Scanning for notable moments..."):
                    st.markdown(ai_helper.conversation_highlights(selected_user, filtered_df))

    elif sub == "People Analysis":
        nav.log_exec("render_ai:people_analysis")
        with _card("Personality Read", "Pick a specific participant (not \"Overall\") to enable this"):
            if selected_user == "Overall":
                st.info("Select a specific user in the sidebar to run a personality read.")
            elif st.button("Analyze Personality", key="ai_personality_btn"):
                with st.spinner("Reading the messaging style..."):
                    st.markdown(ai_helper.personality_analysis(selected_user, filtered_df))

        with _card("Friendship / Dynamic Analysis"):
            people = options["users"]
            if len(people) < 2:
                st.info("Need at least two participants in the chat for this feature.")
            else:
                fc1, fc2 = st.columns(2)
                friend_a = fc1.selectbox("Person A", people, index=0, key="ai_friend_a")
                friend_b = fc2.selectbox(
                    "Person B", people, index=1 if len(people) > 1 else 0, key="ai_friend_b",
                )
                if st.button("Analyze Dynamic", key="ai_friendship_btn"):
                    with st.spinner("Comparing messaging patterns..."):
                        st.markdown(ai_helper.friendship_analysis(filtered_df, friend_a, friend_b))

    st.divider()
    st.caption("AI features analyze your current sidebar-filtered selection and may take a few seconds to respond.")


@st.cache_data(show_spinner=False)
def _export_csv_bytes_cached(_export_df: pd.DataFrame, key: str) -> bytes:
    return utils.to_csv_bytes(_export_df)


@st.cache_data(show_spinner=False)
def _export_excel_bytes_cached(_export_df: pd.DataFrame, key: str) -> bytes:
    return utils.to_excel_bytes(_export_df)


@st.cache_data(show_spinner=False)
def _export_pdf_bytes_cached(_export_df: pd.DataFrame, key: str) -> bytes:
    return utils.to_pdf_bytes(_export_df)


_EXPORT_FORMATS: dict[str, dict] = {
    "CSV": {
        "fn": _export_csv_bytes_cached, "file_name": "shadowtrace_export.csv", "mime": "text/csv",
    },
    "Excel": {
        "fn": _export_excel_bytes_cached, "file_name": "shadowtrace_export.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "PDF": {
        "fn": _export_pdf_bytes_cached, "file_name": "shadowtrace_export.pdf", "mime": "application/pdf",
    },
}


def render_export(filtered_df: pd.DataFrame) -> None:
    """🛠️ Tools — export the current filtered selection as CSV/Excel/PDF.

    Fix 5 (lazy export generation): opening this tab used to eagerly
    serialize the full filtered selection to CSV, Excel, AND PDF on
    every render, whether or not the user ever downloaded any of them
    -- three full passes over the export columns just from opening the
    Tools tab. Now exactly ONE format is generated, only after the user
    picks it and clicks "Generate File". Each format's bytes stay
    cached per the export DataFrame's content (`_export_*_bytes_cached`
    below), so re-generating the same format for the same filtered
    selection is still a cache hit, not a re-serialize.
    """
    nav.log_exec("render_export")
    _page_header("Tools", "Export your filtered conversation data.")

    with _card("Export Your Analysis", f"{len(filtered_df):,} messages based on current filters"):
        export_cols = ['date', 'user', 'message', 'year', 'month', 'day_name', 'hour']
        export_df = filtered_df[export_cols]
        export_key = nlp_helper.content_key(export_df, columns=tuple(export_cols))

        fmt = st.radio("Format", list(_EXPORT_FORMATS.keys()), horizontal=True, key="export_format")

        if st.button("Generate File", key="export_generate_btn", type="primary"):
            spec = _EXPORT_FORMATS[fmt]
            with st.spinner(f"Preparing your {fmt} file..."):
                data = spec["fn"](export_df, export_key)
            st.session_state["_export_ready"] = {
                "format": fmt, "key": export_key, "data": data,
                "file_name": spec["file_name"], "mime": spec["mime"],
            }

        ready = st.session_state.get("_export_ready")
        if ready and ready["format"] == fmt and ready["key"] == export_key:
            st.download_button(
                f"⬇️ Download {fmt}", data=ready["data"],
                file_name=ready["file_name"], mime=ready["mime"],
                width='stretch',
            )

    with _card("Preview", "Exactly what will be exported"):
        st.dataframe(export_df, width='stretch', height=400)


# =============================================================================
# MAIN DISPATCH
# =============================================================================
# By the time this line runs, ui_stage is guaranteed to be "dashboard"
# (both the "landing"/"filtering" and "analyzing" branches above always
# end in st.stop() or st.rerun()). This is a plain if/elif chain (NOT
# st.tabs()) that calls exactly one render_*() function per rerun,
# matching whatever `nav.top_nav()` returns. Every other render_*()
# function simply never gets called — its body never executes, so its
# analytics/NLP/AI calls never run.
with nav.time_block("filtering"):
    filtered_df = get_filtered_df()

if filtered_df.empty:
    st.warning("No messages match the current filters. Try widening them.")
    st.stop()

_main_header_brand()
selected_section = nav.top_nav()

with nav.time_block("section"):
    if selected_section == "Overview":
        render_overview(selected_user, filtered_df)
    elif selected_section == "Activity":
        render_activity(selected_user, filtered_df, options)
    elif selected_section == "People":
        render_users(selected_user, filtered_df, options)
    elif selected_section == "Conversations":
        render_content(selected_user, filtered_df, options)
    elif selected_section == "Insights":
        render_nlp(selected_user, filtered_df)
    elif selected_section == "AI Assistant":
        render_ai(selected_user, filtered_df, options, get_ai_fingerprint())
    elif selected_section == "Tools":
        render_export(filtered_df)

_render_footer()
