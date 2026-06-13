# -*- coding: utf-8 -*-
"""Build thesis_paper.docx — a two-column, IEEE-style thesis paper at repo root.

DOCX is chosen for direct import into Google Docs. Math uses Unicode notation
(editable text, survives import). Figures come from scripts/build_paper_figures.py.
Regenerate with:  python scripts/build_thesis_paper.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "output" / "paper_figs"
OUT = ROOT / "thesis_paper.docx"

doc = Document()

# ---------- page + base style ----------
sec0 = doc.sections[0]
sec0.page_width, sec0.page_height = Inches(8.5), Inches(11)
for s in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec0, s, Inches(0.75))

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(9.5)
style.paragraph_format.space_after = Pt(3)


def set_cols(section, n: int, space: int = 240) -> None:
    cols = section._sectPr.xpath("./w:cols")[0]
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(space))


def P(text: str, *, bold=False, italic=False, size=9.5, align=None, space_after=3):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def H1(text: str):
    P(text, bold=True, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)


def H2(text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(2)


def EQ(text: str, num: int):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text + "        (" + str(num) + ")")
    r.italic = True
    r.font.size = Pt(9.5)
    p.paragraph_format.space_after = Pt(4)


def FIG(path: Path, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(3.2))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.font.size = Pt(8)
    c.paragraph_format.space_after = Pt(6)


def TABLE(header: list[str], rows: list[list[str]], caption: str, widths=None):
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.font.size = Pt(8)
    r.bold = True
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Light Grid Accent 1"
    for j, h in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.text = h
        for par in cell.paragraphs:
            for run in par.runs:
                run.bold = True
                run.font.size = Pt(8)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = t.rows[1 + i].cells[j]
            cell.text = v
            for par in cell.paragraphs:
                for run in par.runs:
                    run.font.size = Pt(8)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ====================================================================
# TITLE BLOCK (single column)
# ====================================================================
P("Grid-Aware Spatio-Temporal Load Shaping for Geo-Distributed Data Centers "
  "via Deep Reinforcement Learning:", bold=True, size=15,
  align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
P("A Cell-Aggregate Study on Google ClusterData 2019", bold=True, size=13,
  align=WD_ALIGN_PARAGRAPH.CENTER, space_after=8)
P("Janusz Gal", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
P("Master's Thesis Draft — generated working paper; numbers correspond to the "
  "multi-seed review campaign (5 seeds/config) on the corrected environment.",
  italic=True, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

P("Abstract — Hyperscale data centers are now grid-scale electrical loads whose "
  "consumption coincides with regional net-demand peaks (the “duck curve”). "
  "We study whether a reinforcement-learning agent can route workload across "
  "geographically distributed data centers — spatially (where) and temporally "
  "(when, for deferrable batch work) — to minimize electricity cost while reducing "
  "the fleet's contribution to grid stress. Following the aggregate load-shaping "
  "paradigm of Google's production Carbon-Intelligent Computing System rather than "
  "job-level scheduling, we model four Borg cells from the Google ClusterData 2019 "
  "trace as proxy data centers: measured per-tier demand curves (service vs. "
  "no-SLO batch, extracted from instance-level usage by priority tier), per-cell "
  "power models calibrated on the companion PowerData2019 measurements "
  "(R² = 0.75–0.80 per cell vs. 0.43 pooled), real EIA-930 net-demand series, and "
  "documented synthetic regional price series (see provenance note, Sec. IV-D). "
  "Across five training seeds per configuration, a continuous-action PPO policy "
  "controlling routing fractions, batch drain rates, and batch placement beats the "
  "grid-unaware status quo by 2.7–12.9%, a foresighted lookahead heuristic by "
  "5.0–14.8%, and the best discrete DQN variant by 2.5–9.9% (winning 19 of 20 "
  "seed-paired comparisons), and sits within 1.1–5.6% of a clairvoyant convex-QP "
  "lower bound — with zero deadline violations and 100% of service demand served. "
  "A three-stage generalization study shows transfer succeeds along the observed "
  "axis (a real 2019→2024 net-demand shift) but fails along an unobserved one "
  "(held-out cells), until domain randomization restores it; and the spatial "
  "advantage survives inter-site movement costs up to ~0.35–0.53× the energy cost "
  "of serving a unit. We additionally contribute three experimentally validated "
  "modeling lessons — deadline penalties must scale with the energy value of "
  "deferred work; capacity-blocked work must queue, not expire; and synthetic "
  "workload generators must conserve the demand-presentation process (requests "
  "≠ usage) — each of which, while uncorrected, inverted the experimental "
  "ranking.", size=9)
P("Index Terms — data centers, demand response, reinforcement learning, workload "
  "scheduling, duck curve, Google cluster trace, load shaping.",
  italic=True, size=9, space_after=10)

# ====================================================================
# TWO-COLUMN BODY
# ====================================================================
body = doc.add_section(WD_SECTION.CONTINUOUS)
set_cols(body, 2)

# ---------------- I. INTRODUCTION ----------------
H1("I. INTRODUCTION")
P("Data centers consumed roughly 415 TWh of electricity in 2024, and the IEA "
  "projects consumption to more than double to ~950 TWh — about 3% of global "
  "electricity demand — by 2030 [28]; the geographic concentration of "
  "hyperscale campuses turns individual operators into grid-scale loads. Simultaneously, solar-heavy grids exhibit the duck "
  "curve: midday renewable generation depresses net demand, followed by a steep "
  "evening ramp during which prices spike and grid stress peaks. A data-center "
  "fleet that concentrates its draw during these ramps contributes directly to "
  "system peaks; a fleet that shifts flexible work across locations and hours "
  "can instead smooth them. Google's production Carbon-Intelligent Computing "
  "System (CICS) demonstrates that this lever is real at hyperscale [3]. Open "
  "RL environments for sustainable data-center scheduling now exist — "
  "SustainDC for within-DC multi-agent control [26] and SustainCluster for "
  "geo-distributed per-task dispatch [27] — but none provides measured "
  "per-tier (flexible vs. inflexible) demand decomposition of a hyperscale "
  "trace, power models calibrated on companion power measurements, or a grid "
  "net-demand (duck-curve) objective; those are the gaps this thesis fills.")
P("This thesis asks: can a model-free reinforcement-learning agent, observing "
  "only per-site demand, prices, and grid net demand, learn a joint spatial-"
  "temporal load-shaping policy that beats both the grid-unaware status quo and "
  "foresighted scheduling heuristics? We answer affirmatively, on an environment "
  "built end-to-end from measured data, and we document — unusually, as a "
  "first-class contribution — the modeling errors we made on the way, each of "
  "which inverted the experimental ranking until found and fixed.")
P("Contributions. (C1) A reproducible Gymnasium environment for multi-data-center "
  "grid-aware load shaping at cell-aggregate granularity, grounded in Google "
  "ClusterData 2019 [1], PowerData2019 [2], EIA-930 net demand, and documented "
  "synthetic regional prices. (C2) Ground-truth measured per-tier demand "
  "decomposition (SLO service vs. no-SLO batch) of four trace cells, with the "
  "finding that resource requests overestimate the deferrable usage share by "
  "3–5×. (C3) A continuous-action PPO formulation (routing + drain + batch "
  "placement) that, across five seeds per configuration, wins every evaluated "
  "configuration against heuristic, DQN, and status-quo baselines with zero "
  "deadline violations, and lands within 1.1–5.6% of a clairvoyant QP lower "
  "bound. (C4) Identification of marginal-power (slope) arbitrage as an emergent "
  "routing strategy enabled by per-cell power calibration. (C5) A three-stage "
  "generalization result establishing that transfer tracks observability, with "
  "domain randomization closing the unobserved axis. (C6) Three transferable "
  "modeling lessons for deferral-with-deadlines simulators, each validated by "
  "ranking inversion.")

# ---------------- II. RELATED WORK ----------------
H1("II. RELATED WORK AND METHODOLOGICAL LINEAGE")
P("This work deliberately assembles its methodology from prior art; this section "
  "states precisely what is taken from where.")
H2("A. The trace and its semantics")
P("ClusterData 2019, documented by Tirmazi et al. [1], covers eight Borg cells "
  "for May 2019 (~96k machines). We adopt three of its definitions wholesale: "
  "the cell as the natural management unit (§2 of [1]); priority tiers, from "
  "which we define deferrable work as the union of the two no-SLO tiers — "
  "free (priority ≤ 99) and best-effort batch (100–115, per the trace "
  "documentation [22], which corrects the 110–115 range mistakenly reported "
  "in [1]) — production work (120–359) is explicitly protected by eviction "
  "of lower tiers and is never deferred; and Normalized Compute Units. The companion PowerData2019 [2] "
  "provides measured per-PDU power utilization, used here for calibration. "
  "Tirmazi's heavy-tail observation (top 1% of jobs consume >99% of resources) "
  "informs our distribution fits; notably we find the tail does not survive "
  "aggregation (Sec. VII).")
H2("B. Workload-generation lineage")
P("Our synthetic batch generator descends directly from Da Costa, Grange & "
  "De Courchelle [5], instantiated as the scipy-based Listing 1 of Grange et "
  "al. [4], in the Feitelson workload-modeling tradition [21]. We reproduce "
  "Listing 1 verbatim (scripts/grange_generator.py recovers their published "
  "lognormal s = 1.634, scale = 447), then extend it: distributions are re-fit "
  "to the 2019 trace under the no-SLO definition, selected by minimum "
  "Kolmogorov–Smirnov distance (the p-value saturates at n ≈ 10⁵), "
  "with discrete negative-binomial task counts. Grange's SLA-flexibility knob "
  "becomes our deadline model, Eq. (5). Critically, in the final environment the "
  "generator is demoted to sensitivity analysis: measured per-tier curves are "
  "the primary input, after we found single-timestep job pulses inflated batch "
  "burstiness by two orders of magnitude (Sec. VII).")
H2("C. Geographic load balancing for electricity cost")
P("Spatial electricity-price arbitrage across internet-scale systems predates "
  "the renewable-aware literature: Qureshi et al. quantified the savings from "
  "routing request load toward cheap electricity markets [23]; Rao et al. "
  "formalized cost minimization for distributed data centers across multiple "
  "electricity markets [24]; and Liu, Lin, Wierman, Low and Andrew gave the "
  "optimization treatment of geographical load balancing with renewable "
  "supply, characterizing when 'follow the renewables' routing is optimal "
  "[25]. Our spatial lever is a direct descendant of this lineage — with two "
  "displacements: the objective targets grid net demand (the duck-curve "
  "quantity) rather than price alone, and the routed quantity is measured "
  "aggregate tier demand rather than request traffic.")
H2("D. Single-DC renewable-aware scheduling")
P("The deferral-with-deadlines pattern originates in single-DC work: GreenSlot "
  "[10] delays jobs toward predicted cheap/green slots; Grange et al. [4] add "
  "due-date constraints and an infrastructure-agnostic objective signal (49% "
  "brown-energy reduction); Xu et al. [7] split workloads into brownout-able "
  "interactive and deferrable batch; Haghshenas et al. [8] schedule heterogeneous "
  "workloads against rate structures; Liu et al. [9] forecast-then-plan. We "
  "benchmark against this lineage at the level of objectives and effects — cost, "
  "peak contribution, deferral value — not mechanism: our Trough-Slot baseline "
  "retargets GreenSlot's slot valuation from solar supply to grid net demand, "
  "and our service/batch split operationalizes Xu's taxonomy via measured tiers.")
H2("E. Aggregate load shaping at hyperscale")
P("Radovanović et al. [3] describe CICS, which shapes aggregate cluster "
  "load via day-ahead Virtual Capacity Curves and explicitly supersedes "
  "job-level deadline optimization (“aggregate cluster-specific resource "
  "demand forecasts … rather than stylized models for job-level resource "
  "demand modeling”). Our environment is a CICS-shaped formulation made "
  "reproducible: aggregate flexible/inflexible demand curves per cluster, "
  "power models trained separately per cluster, and both spatial and temporal "
  "shifting — retargeted from carbon to grid demand smoothing, on public data.")
H2("F. Reinforcement learning and open environments")
P("RL for DC energy management is surveyed in [15]; DeepEE [16] jointly "
  "schedules jobs and cooling with DRL (cooling is excluded from our scope). "
  "Closest to our setting, CFWS [6] applies DQN with a flattened-index action "
  "encoding to VM migration across four geo-distributed DCs. We do not "
  "reimplement CFWS (it operates per-VM/per-PM); instead we port its action-"
  "encoding philosophy to a 48-action variant and benchmark it, alongside a "
  "759-action routing grid, against continuous-action PPO [18] (DQN [19]; "
  "implementations via Stable-Baselines3 [20]). Energy-model form follows the "
  "linear idle+slope server model surveyed in [14] and canonicalized by Fan et "
  "al. [17]; the demand-response framing follows [13]; broader green-DC context "
  "in [11], geo-distributed scheduling in [12].")
P("Open benchmark environments for this problem class have recently emerged. "
  "SustainDC [26] provides multi-agent Gymnasium environments for control "
  "*within* a data center (workload shifting, cooling, battery). SustainCluster "
  "[27] is its geo-distributed successor: a centralized scheduler dispatches or "
  "defers *individual tasks* of the Alibaba 2020 GPU trace across 20+ global "
  "locations every 15 minutes, optimizing energy cost, carbon, SLA, and "
  "per-GB transmission overheads. Our environment is complementary on three "
  "axes: granularity (CICS-style measured aggregate per-tier curves rather "
  "than per-task dispatch), data grounding (Google ClusterData 2019 with "
  "PowerData2019-calibrated per-cell power, rather than Alibaba GPU jobs with "
  "carbon-intensity feeds), and objective (grid net-demand peak contribution "
  "rather than carbon). We deliberately do not run our experiments inside "
  "SustainCluster: answering our question there would require replacing both "
  "its objective (carbon → grid net demand) and its action model (per-task "
  "dispatch → aggregate tier control, the granularity Sec. II-E argues for), "
  "while abandoning the Google-trace data contributions — at which point the "
  "environment would no longer be SustainCluster. The two environments are "
  "complementary benchmarks; SustainCluster's transmission-cost model is the "
  "natural template for the movement-cost sensitivity analysis of Sec. VIII, "
  "and cross-validation of our policies inside it is listed as future work.")

# ---------------- III. SYSTEM MODEL ----------------
H1("III. SYSTEM MODEL")
P("N = 4 data centers indexed by i operate over T = 8,917 five-minute steps "
  "(Δ = 300 s, Δh = 1/12 h). Site i has measured aggregate demand "
  "curve wᵢ,ₜ ∈ [0,1] (fraction of cell capacity), decomposed by "
  "measured priority tier into non-deferrable service demand vᵢ,ₜ and "
  "deferrable batch arrivals aᵢ,ₜ with v + a = w exactly. Site inputs "
  "further include electricity price πᵢ,ₜ ($/kWh), normalized grid "
  "net demand dᵢ,ₜ ∈ [0,1], and capacity κᵢ from fleet "
  "machine data.")
H2("A. Power and cost")
P("Each site has a per-cell calibrated linear power model:")
EQ("Pᵢ(u) = Pᵢⁱᵈˡᵉ + sᵢ · u", 1)
EQ("gᵢ,ₜ = Pᵢ(uᵢ,ₜ) · R,   R = 100 MW", 2)
P("where uᵢ,ₜ is served CPU and g the grid draw (sites are pure grid "
  "loads; no on-site generation). Energy cost and the quadratic, net-demand-"
  "weighted peak-contribution penalty are")
EQ("Eₜ = Σᵢ πᵢ,ₜ · gᵢ,ₜ · 1000 · Δh", 3)
EQ("Φₜ = α · Σᵢ gᵢ,ₜ² · dᵢ,ₜ,   α = 0.015", 4)
P("The quadratic form penalizes concentration; the d-weighting makes draw "
  "near the duck-curve neck expensive and slack-hour draw nearly free [13].")
H2("B. Deferrable batch dynamics")
P("Each site maintains a pool Qᵢ,ₜ of batch work with per-entry "
  "deadlines. Arrivals enter with deadline offset from the fitted mean duration "
  "μᵢ and flexibility factor φ = 1 (after [4]):")
EQ("τ = t + ⌈ μᵢ (1 + φ) / Δ ⌉", 5)
EQ("Qᵢ,ₜ₊₁ = Qᵢ,ₜ + aᵢ,ₜ − Sᵢ,ₜ − Xᵢ,ₜ", 6)
P("where S is batch actually served and X expired work. Two semantics are "
  "essential and were validated by invariant (Sec. V-C): the drain action is an "
  "intended release — only served work leaves the pool — and capacity-blocked "
  "work retains its original deadline (it queues, as Borg's batch scheduler "
  "does [1], rather than expiring on a one-step fuse).")
H2("C. Action, serving, and reward")
P("The agent outputs a ∈ [−3,3]³ᴺ, decoded as service routing "
  "fractions f = softmax(a₁:ₙ), drain rates δ = σ(aₙ₊₁:₂ₙ), "
  "and batch placement h = softmax(a₂ₙ₊₁:₃ₙ) — batch "
  "work, having no latency SLO, may execute anywhere:")
EQ("uᵢ,ₜ = min( fᵢ Dₜ + Bᵢ,ₜ + hᵢ Σⱼ δⱼ Qⱼ,ₜ ,  κᵢ )", 7)
P("with Dₜ total service demand, B backlog (unserved service, penalized), "
  "and service strictly prioritized over batch. The ±3 action bound (not the "
  "SB3 default ±1) matters: SB3 clips actions to the action box before the "
  "decode, so ±1 would cap every routing share to [4.3%, 71%] and every drain "
  "rate to [27%, 73%], making full concentration and multi-hour holding "
  "impossible by construction; ±3 restores parity with the discrete baselines "
  "(shares to ~98.5%). The reward is")
EQ("rₜ = −[ Eₜ + Φₜ + λ_b Σ B + λ_κ Σ max(0, u−κ) + w Σ X ]", 8)
P("with capacity weight λ_κ = 5, deadline weight w = 250, and service-"
  "backlog weight λ_b = 25. Both w and λ_b are calibrated to the energy value "
  "of the work they protect (Sec. VII, L1): w sits safely above both the ~$150 "
  "cost of serving a deferred unit and the analytic ranking-inversion point "
  "w ≈ 183, and λ_b is set so that parking SLO service in the backlog to dodge "
  "a price spike is never profitable (at the SB3-typical λ_b = 1.5 a three-hour "
  "hold costs only ~$54 against ~$180 of arbitrage, an exploit the audit in "
  "Sec. VI confirms is closed). The demand-smoothing metric reported alongside "
  "cost is the fleet load factor LF = mean(g)/max(g).")

# ---------------- IV. DATA ----------------
H1("IV. DATA AND CALIBRATION")
H2("A. Cells as proxy data centers")
P("Cells a–d serve as four proxy DCs. Cell locations in the trace are "
  "anonymized; the geographic assignment is a modeling choice (US scenario: a "
  "CAISO-exposed Western DC + MISO/Southern/Duke territories; Global scenario: "
  "US-West/US-Central/NL/Singapore). Tirmazi documents considerable inter-cell "
  "workload variation but no geography. The normalized curve provides the "
  "demand shape; R = 100 MW sets hyperscale magnitude (a real cell is ~3–5 MW, "
  "invisible to a regional grid). Sensitivity to R is future work.")
H2("B. Measured per-tier demand")
P("The service/batch split is measured, not modeled: instance_usage is joined "
  "to collection priority and aggregated per 5-minute bucket into service (SLO "
  "tiers) and batch (no-SLO tiers, priority ≤ 115) curves with v + a = w to "
  "machine precision (Fig. 2). Measured deferrable usage shares are 16.6% (a), "
  "15.7% (b), 22.3% (c), 26.4% (d) — consistent with the ~20%-of-capacity "
  "best-effort average reported for the trace [1]. Request-weighted proxies "
  "(job counts × requested CPU) suggest 63–85%, an overestimate of 3–5×, "
  "consistent with Borg's over-allocation of best-effort tiers. Requests "
  "measure intent; usage measures schedulable reality.")
FIG(FIGS / "fig2_tiers.png",
    "Fig. 2. Measured per-tier decomposition of cell b (3 of 31 days): "
    "service + batch equals the measured aggregate at every step.")
H2("C. Per-cell power calibration")
P("Joining hourly PowerData2019 measured power with aggregate CPU yields 2,980 "
  "(cell, hour) samples. A pooled linear fit gives idle 0.479, slope 0.444, "
  "R² = 0.43 — but the residual is between-cell heterogeneity, not noise: "
  "per-cell fits reach R² = 0.75–0.80 (Fig. 3, Table II), and adding "
  "memory as a regressor contributes only +0.03. The environment therefore uses "
  "per-cell models, mirroring CICS's per-cluster power models [3]. The cells "
  "span a meaningful proportionality range: cell d (idle 0.38, slope 0.57) is "
  "far more energy-proportional than cell a (0.53, 0.34) — a real routing "
  "signal (Sec. VI-B).")
TABLE(["Cell", "idle", "slope", "R²"],
      [["a", "0.528", "0.340", "0.79"],
       ["b", "0.548", "0.376", "0.75"],
       ["c", "0.440", "0.526", "0.76"],
       ["d", "0.382", "0.570", "0.80"],
       ["pooled", "0.479", "0.444", "0.43"]],
      "TABLE II. POWER MODEL CALIBRATION (POWERDATA2019)",
      widths=[0.6, 0.7, 0.7, 0.6])
FIG(FIGS / "fig3_power.png",
    "Fig. 3. Per-cell power calibration: four distinct idle/slope lines; the "
    "pooled fit (dashed) blurs them into R² = 0.43.")
H2("D. Market-data provenance (prices and net demand)")
P("We state the price and net-demand provenance precisely, since it bounds the "
  "claims. NET DEMAND is real: hourly EIA-930 series (demand minus utility-"
  "scale wind+solar) for the assigned US balancing authorities, normalized to "
  "[0,1] by each region's window peak; the Global scenario's EU/Singapore net "
  "demand is synthesized where EIA-930 does not apply. PRICES, by contrast, are "
  "DOCUMENTED SYNTHETIC: a calibrated diurnal model anchored to plausible "
  "May-2019 regional wholesale averages, NOT real locational marginal prices. "
  "Real hourly LMP is not uniformly obtainable for this footprint — two US "
  "scenario regions (Southern Co./GA, Duke/SC) are vertically integrated "
  "utilities with no public wholesale market, and the EIA API exposes no hourly "
  "price route — so a fully-real price series is not currently available. This "
  "is a known limitation we intend to resolve (real CAISO/MISO LMP plus "
  "documented proxies for the non-ISO territories) in future work. Two points "
  "bound its impact. First, the relative comparison is unaffected: every policy "
  "is scored against the SAME price series, so any price bias cancels in the "
  "differences between policies, which are what we report. Second, the price "
  "DYNAMICS the agent exploits (diurnal spread, cross-region differences) are "
  "structurally realistic even if the absolute levels are modeled. The "
  "generalization study (Sec. VI-G) additionally shifts net demand to real "
  "EIA-930 May-2024 data, testing transfer on the one fully-real market signal.")

# ---------------- V. EXPERIMENTAL SETUP ----------------
H1("V. EXPERIMENTAL SETUP")
H2("A. Agents and baselines")
P("PPO (Stable-Baselines3, MLP 128×128, lr 3e-4, 500k steps) acts in the "
  "continuous 3N-dimensional space. Its observation includes per-DC dynamic "
  "state (demand, price, net demand, pool/backlog) AND static site context "
  "(per-DC idle power, slope, capacity, deferrable fraction) — the latter "
  "added so that per-site heterogeneity is exploitable as a function of "
  "observed parameters rather than memorized by slot (Sec. VI-G). Two DQN "
  "variants share hyperparameters (MLP 256×256, replay 100k, target update "
  "1k): a 759-action routing grid, and a 48-action CFWS-style flattened index "
  "decoding to (source DC, destination DC, drain level) [6]. Nine heuristics "
  "span the policy space, including Round Robin, price-chasing and slack-grid "
  "concentrators, defer-to-trough rules, a GreenSlot-style Trough-Slot "
  "Lookahead foresighted heuristic with privileged 3-hour net-demand foresight "
  "[10], and the no-optimization Status Quo (serve locally, immediately) — the "
  "counterfactual of a CICS-style layer switched off.")
H2("B. Configurations and protocol")
P("{US, Global} × {spatial-only, batch = spatial + temporal}. Each (algorithm, "
  "configuration) is trained with FIVE seeds; we report mean ± std and paired "
  "PPO-vs-best-DQN statistics (sign test, Wilcoxon, bootstrap CI). All policies "
  "are then evaluated deterministically on the identical full-trace episode.")
H2("C. Validation invariant")
P("By construction service + batch = measured demand, so a serve-everything-"
  "now policy must reproduce the spatial-only run exactly. This holds to the "
  "dollar: Status Quo scores $9,847,536 in both US modes with zero expiry — the "
  "batch machinery is demand-neutral, and any batch-mode difference between "
  "policies is scheduling, not artifact. This invariant caught two of the three "
  "modeling errors of Sec. VII.")

# ---------------- VI. RESULTS ----------------
H1("VI. RESULTS")
TABLE(["Config", "SQ", "Trough", "DQN", "PPO (±sd)", "ΔSQ", "QP", "gap"],
      [["US spatial", "9.85", "11.23", "9.81", "9.57±.03", "2.8%", "9.46", "1.1%"],
       ["US batch", "9.85", "10.09", "9.89", "9.59±.08", "2.7%", "9.44", "1.5%"],
       ["Glob spatial", "14.06", "14.32", "13.35", "12.62±.25", "10.3%", "11.95", "5.6%"],
       ["Glob batch", "14.06", "13.76", "13.46", "12.25±.14", "12.9%", "11.92", "2.7%"]],
      "TABLE I. EPISODE COST (M$), 5-SEED MEAN. ΔSQ = PPO SAVINGS VS STATUS "
      "QUO; QP = CLAIRVOYANT OPTIMUM; gap = PPO ABOVE QP.",
      widths=[0.92, 0.5, 0.55, 0.5, 0.72, 0.5, 0.5, 0.46])
FIG(FIGS / "fig1_results.png",
    "Fig. 1. Five-seed mean episode cost (PPO error bars = ±1 sd); red dashed "
    "line is the clairvoyant QP lower bound. PPO savings vs. status quo "
    "annotated.")
H2("A. Headline")
P("Across five seeds per configuration, PPO is the best policy — learned or "
  "heuristic — in all four: 2.7–12.9% below the grid-unaware status quo, "
  "5.0–14.8% below the foresighted Trough-Slot heuristic, and 2.5–9.9% below "
  "the best DQN variant. PPO beats the best DQN in 19 of 20 seed-paired "
  "comparisons (sign/Wilcoxon p = 0.031 in three of four configs; the "
  "exception is Global spatial-only, 4/5, p = 0.19). Crucially, PPO lands "
  "within 1.1–5.6% of a clairvoyant convex-QP lower bound (Sec. VI-F) — so "
  "little headroom remains to ANY policy, causal or not. Savings scale with "
  "exploitable structure: modest under US-only diversity, large under global "
  "price/timezone spread. PPO also serves 100% of service demand with zero "
  "deadline violations (Sec. VI-E) and the flattest, lowest fleet draw "
  "(Fig. 4).")
FIG(FIGS / "fig4_profile.png",
    "Fig. 4. Fleet power profile, US batch (first 4 days): PPO serves the same "
    "work at lower, flatter draw.")
H2("B. Emergent slope arbitrage")
P("Per-cell power calibration converts the low-diversity US scenario from "
  "“nothing to learn” (under pooled power, all DCs are energetically identical "
  "and near-uniform routing is optimal) into a real optimization: idle power is "
  "sunk, so each marginal unit of CPU is cheapest where the slope is lowest. "
  "PPO inverts the load distribution relative to the do-nothing policies — "
  "toward low-slope cells a/b, away from high-slope c/d — worth +2.8% over the "
  "status quo with no spatial price diversity at all. No baseline encodes this "
  "strategy. In the Global scenario, slope arbitrage compounds with price "
  "arbitrage: PPO routes aggressively away from high-priced Singapore toward "
  "cheap, low-slope US capacity.")
H2("C. The foresighted heuristic loses everywhere")
P("Trough-Slot Lookahead holds privileged 3-hour future net-demand information "
  "yet loses every configuration by 5.0–14.8% vs. PPO — and in the US it is the "
  "single most expensive policy, worse even than the do-nothing status quo. Two "
  "structural flaws cost it: its slack-grid routing rule CONCENTRATES load, "
  "raising the fleet peak that the quadratic penalty punishes (LF ≈ 0.86 vs. "
  "PPO's ≈ 0.96), and it is blind to per-cell power (it optimizes against net "
  "demand only, while PPO additionally arbitrages each cell's slope). Foresight "
  "does not compensate for optimizing the wrong surface.")
H2("D. The temporal lever, honestly sized")
P("At the measured deferrable fractions (16–26% of usage), batch deferral adds "
  "+2.7 points over the spatial-only result in the Global scenario "
  "(+10.3% → +12.9% vs. status quo) but is essentially neutral in the US "
  "(+2.8% → +2.7%), where there is little price or timezone diversity for "
  "temporal shifting to exploit. The temporal lever is real but secondary to "
  "spatial routing, and it is scenario-dependent. (Earlier drafts' inflated "
  "+6% estimates were artifacts L1–L3, Sec. VII.)")
H2("E. Deadline and backlog audit")
P("Because every policy is scored on the shaped objective (Eq. 8), we audit "
  "that PPO's savings are genuine scheduling, not penalty-dodging. Across all "
  "ten PPO batch-mode seeds: service served / demand = 1.0000, terminal "
  "backlog = 0, terminal batch pool ≤ 0.95 units, ZERO expired batch, peak "
  "transient backlog ≤ 3.9 units, and backlog penalty ≤ 1% of total cost. PPO "
  "is not parking interactive work to avoid price spikes (the λ_b = 25 "
  "calibration, Sec. III-C, makes that unprofitable) nor leaking work past the "
  "finite horizon — it serves everything, on time, more cheaply.")
H2("F. Optimality gap")
P("The environment is convex in the serving decisions (linear power and energy "
  "cost, convex-quadratic peak penalty, linear queue/backlog dynamics), so a "
  "clairvoyant planner with full-episode foresight and fluid allocation solves "
  "a quadratic program whose optimum LOWER-BOUNDS any policy. We assemble it "
  "directly for the Clarabel solver (per-origin-stream cumulative deadline "
  "constraints are exact; validated against cvxpy at 10⁻⁹) and solve all four "
  "configs in 1–6 s. PPO's gap to this bound is 1.1% (US spatial), 1.5% (US "
  "batch), 5.6% (Global spatial), 2.7% (Global batch). The bound is generous — "
  "it has perfect foresight and no causality or action-parameterization "
  "constraints — so the true gap to the best ACHIEVABLE causal policy is "
  "smaller still. A useful by-product: the QP's spatial-only and batch optima "
  "differ by only ~0.2%, confirming the intrinsic value of temporal "
  "flexibility at these deferrable fractions is small (Sec. VI-D).")
H2("G. Generalization: transfer tracks observability")
P("We test the frozen policies (no retraining) on two distribution shifts "
  "(Fig. 6). On an UNOBSERVED shift — held-out cells e–h, with workload, tier "
  "mix, and per-cell power the policy never saw — context-aware PPO fails "
  "where its edge is purely spatial (US spatial −19%, Global spatial −27% vs. "
  "the held-out status quo): the policy had learned per-site routing keyed to "
  "static parameters that, although now in the observation, never VARIED "
  "during training and so carried no gradient signal. Training-time domain "
  "randomization (permuting compute bundles across market slots and resampling "
  "power parameters within the measured 8-cell range) fixes this, restoring "
  "positive transfer in three of four configs (+4.0–5.2%); Global spatial-only, "
  "whose price-concentration strategy has no deferral pool to absorb mistakes "
  "on unseen capacity, stays brittle — and its batch counterpart, which does, "
  "transfers at +5.2%. On an OBSERVED shift — net demand swapped to real "
  "EIA-930 May-2024 (a genuine 2019→2024 duck-curve change), prices held "
  "fixed — the SAME frozen policies transfer near-perfectly (+3.1–13.2%, within "
  "~0.5 point of their on-training result). Generalization succeeds along the "
  "axis the policy can observe and fails along the one it cannot; this both "
  "answers the calendar-memorization concern for the demand signal and "
  "prescribes the fix (observe + randomize static context).")
FIG(FIGS / "fig6_generalization.png",
    "Fig. 6. Held-out transfer. Red: unobserved cell shift breaks context-only "
    "policies. Green: domain randomization restores it. Blue: the observed "
    "net-demand-year shift transfers without any special treatment.")
H2("H. Robustness to movement cost")
P("The spatial lever assumes free inter-site movement; we test how much "
  "survives a per-unit cost on work routed away from its home cell (the "
  "SustainCluster transmission-cost idea [27]). Charging existing policies "
  "post-hoc — a conservative bound, since a movement-aware policy would route "
  "less and recover more — the Global advantage is robust: it survives until "
  "movement costs reach $219/unit (Global spatial) and $331/unit (Global "
  "batch), i.e. 0.35× and 0.53× the energy cost of serving a unit, still "
  "returning +7.9% and +11.0% at a substantial $50/unit (Fig. 7). The US "
  "margin is thinner (break-even ~0.14× unit energy), honestly reflecting its "
  "smaller spatial diversity. The 10–13% Global savings are thus not an "
  "artifact of free fungibility.")
FIG(FIGS / "fig7_movement_cost.png",
    "Fig. 7. PPO savings vs. per-unit inter-site movement cost (symlog x). "
    "Global routing keeps paying well past realistic egress prices.")

# ---------------- VII. LESSONS ----------------
H1("VII. LESSONS LEARNED AND THREATS TO VALIDITY")
P("Three modeling errors each inverted the experimental ranking while present; "
  "we report them as first-class results because any deferral-with-deadlines "
  "simulator can reproduce them.")
P("L1 — Deadline penalties must scale with the value of deferred work. A "
  "fixed penalty of 2 made expiring a unit ~75× cheaper than serving it "
  "(~$150 of energy); the cost-optimal policy was to discard batch, and "
  "aggressive-expiry heuristics beat PPO. Because the penalty is linear in "
  "expiry, any rollout can be re-ranked analytically as cost(w) = "
  "(cost − w₀·x) + w·x (exact for a fixed rollout; retrained agents change "
  "behavior with w); the ranking inverts at w ≈ 183. We set w = 250, safely "
  "above both the ~$150 serving cost and that inversion point.")
P("L2 — Capacity-blocked work must queue, not expire. Re-queueing blocked "
  "batch with a one-step deadline manufactured a policy-independent expiry "
  "floor (~5,059 units, identical across scenarios — the tell), contradicting "
  "Borg's queued batch semantics [1]. Preserving original deadlines collapsed "
  "Status Quo expiry by 85% immediately and to zero in the final environment, "
  "and eliminated a spurious “concentrators win by dumping” effect.")
P("L3 — Synthetic generators must conserve the demand-presentation process. "
  "Injecting each sampled job's whole demand at its arrival timestep produced "
  "batch curves with peak/mean ≈ 195 vs. ≈ 1.24 in the measured trace: "
  "jobs present rate over duration, and concurrency smooths the aggregate. "
  "Corollary (L3b): requests ≠ usage — request-weighted tier shares "
  "overestimated the deferrable fraction 3–5×. A one-line shape "
  "check (generated vs. source peak/mean) catches both.")
P("Threats to validity. (i) PRICES are documented synthetic, not real LMP "
  "(Sec. IV-D) — the most significant limitation; relative comparisons are "
  "unaffected (shared series) and net demand is real, but absolute price levels "
  "are modeled, and resolving this with real ISO prices is primary future work. "
  "(ii) Cell locations are anonymized; the geographic assignment is a modeling "
  "choice justified by documented inter-cell heterogeneity [1]. (iii) The 100 "
  "MW magnitude bridges cell scale to grid relevance; results should be swept "
  "over R, and a 300 MW fleet would plausibly move LMPs (the price-taker "
  "assumption). (iv) The measured tier curves make episodes deterministic; "
  "Sec. VI-G addresses the resulting memorization concern directly and shows "
  "transfer along observed axes. (v) May is the deepest duck-curve month in "
  "CAISO; seasonal generalization is untested. (vi) Cooling and PUE are "
  "excluded by scope; a multiplicative PUE would scale, not reorder, results. "
  "(vii) The linear power model is calibrated on u ≈ 0.1–0.7 and omits "
  "memory/IO effects (+0.03 R² at most); a policy pushing utilization outside "
  "that range extrapolates. (viii) Five seeds and one hyperparameter set per "
  "algorithm; DQN's higher seed variance (std up to ±$0.9M vs. PPO's ≤$0.25M) "
  "is itself reported as a stability finding.")

# ---------------- VIII. FUTURE WORK ----------------
H1("VIII. FURTHER TOPICS OF CONSIDERATION")
P("(1) REAL PRICES: replace the synthetic price series with real CAISO/MISO "
  "LMP (and documented proxies for the non-ISO territories) — the highest-"
  "priority item, since it is the main provenance limitation (Sec. IV-D). "
  "(2) Sensitivity sweeps: deferrable fraction (via the retained generator), "
  "rated power R, flexibility factor φ, peak weight α; and seasonal "
  "generalization beyond May. (3) Carbon objective: swap or add marginal "
  "carbon-intensity signals to compare against CICS's objective directly [3]. "
  "(4) Forecast features: day-ahead price and net-demand forecasts as "
  "observations, quantifying the residual value of explicit foresight. "
  "(5) Constrained RL: enforce zero deadline violations as a hard constraint "
  "(e.g., Lagrangian PPO) rather than a penalty. (6) Hybrid granularity: a "
  "job-level inner scheduler below the aggregate outer policy, bridging to "
  "the CFWS setting [6]. (7) Storage and demand response: batteries and "
  "market participation extend the action space naturally [13]. (8) DQN "
  "stabilization: distributional/double variants and action-space curricula "
  "to separate encoding effects from optimizer fragility. (9) Multi-objective "
  "Pareto analysis of cost vs. peak contribution vs. completion latency. "
  "(10) Transfer to other public traces — e.g., the Alibaba 2020 GPU trace "
  "already packaged by SustainCluster [27] — to test trace dependence of the "
  "slope-arbitrage finding, and cross-validation of our policies inside the "
  "SustainCluster environment.")

# ---------------- IX. CONCLUSION ----------------
H1("IX. CONCLUSION")
P("On an environment assembled largely from measured public data — per-tier "
  "demand from ClusterData 2019, per-cell power from PowerData2019, real "
  "EIA-930 net demand, and documented synthetic prices — a continuous-action "
  "PPO agent learns grid-aware spatio-temporal load shaping that, across five "
  "seeds per configuration, beats the grid-unaware status quo by 2.7–12.9%, a "
  "foresighted heuristic by 5.0–14.8%, and discrete DQN variants by 2.5–9.9% "
  "(19 of 20 seed comparisons) in every configuration, landing within 1.1–5.6% "
  "of a clairvoyant QP bound with zero deadline violations and a flatter, lower "
  "fleet profile. The savings come from continuously exploitable structure — "
  "price spreads and per-cell marginal-power (slope) arbitrage — and they "
  "transfer to unseen market conditions along the axes the policy observes, "
  "with domain randomization extending transfer to unseen workloads. Equally "
  "important, the path to these numbers required finding and fixing several "
  "modeling errors that each inverted the ranking, and an honest accounting of "
  "what is measured (demand, power, net demand) versus modeled (prices); we "
  "offer the corrected environment, the validation invariant, the QP bound, and "
  "the lessons themselves as contributions to reproducible research in "
  "sustainable computing.")

# ---------------- REFERENCES ----------------
H1("REFERENCES")
refs = [
 "M. Tirmazi, A. Barker, N. Deng, M. E. Haque, Z. G. Qin, S. Hand, M. Harchol-Balter, and J. Wilkes, “Borg: the Next Generation,” in Proc. EuroSys, 2020.",
 "V. Sakalkar, V. Kontorinis, D. Landhuis, S. Li, D. De Ronde, T. Blooming, A. Ramesh, J. Kennedy, C. Malone, J. Clidaras, and P. Ranganathan, “Data Center Power Oversubscription with a Medium Voltage Power Plane and Priority-Aware Capping,” in Proc. ASPLOS, 2020.",
 "A. Radovanović, R. Koningstein, I. Schneider, B. Chen, A. Duarte, B. Roy, D. Xiao, M. Haridasan, P. Hung, N. Care, S. Talukdar, E. Mullen, K. Smith, M. Cottman, and W. Cirne, “Carbon-Aware Computing for Datacenters,” IEEE Trans. Power Systems, vol. 38, no. 2, 2023 (arXiv:2106.11750, 2021).",
 "L. Grange, G. Da Costa, and P. Stolf, “Green IT scheduling for data center powered with renewable energy,” Future Generation Computer Systems, vol. 86, pp. 99–120, 2018.",
 "G. Da Costa, L. Grange, and I. De Courchelle, “Modeling and generating large-scale Google-like workload,” in Proc. Int. Green and Sustainable Computing Conf. (IGSC), 2016.",
 "D. Zhao, J.-t. Zhou, and K. Li, “CFWS: DRL-Based Framework for Energy Cost and Carbon Footprint Optimization in Cloud Data Centers,” IEEE Trans. Sustainable Computing, vol. 10, no. 1, pp. 95–107, Jan./Feb. 2025.",
 "M. Xu, A. N. Toosi, and R. Buyya, “A Self-Adaptive Approach for Managing Applications and Harnessing Renewable Energy for Sustainable Cloud Computing,” IEEE Trans. Sustainable Computing, 2020.",
 "K. Haghshenas, A. Taheri, M. Goudarzi, and S. Mohammadi, “Infrastructure-Aware Heterogeneous-Workloads Scheduling for Data Center Energy Cost Minimization,” IEEE Trans. Cloud Computing, 2022.",
 "Z. Liu, Y. Chen, C. Bash, A. Wierman, D. Gmach, Z. Wang, M. Marwah, and C. Hyser, “Renewable and Cooling Aware Workload Management for Sustainable Data Centers,” in Proc. ACM SIGMETRICS, 2012.",
 "Í. Goiri, K. Le, M. E. Haque, R. Beauchea, T. D. Nguyen, J. Guitart, J. Torres, and R. Bianchini, “GreenSlot: Scheduling Energy Consumption in Green Datacenters,” in Proc. SC, 2011.",
 "W. Lin et al., “A systematic review of green-aware management techniques for sustainable data center,” Sustainable Computing: Informatics and Systems, 2024.",
 "Y. Wu et al., “Task Scheduling in Geo-Distributed Computing: A Survey,” arXiv:2501.15504, 2025.",
 "T. L. Vasques, P. Moura, and A. de Almeida, “A review on energy efficiency and demand response with focus on small and medium data centers,” Energy Efficiency, vol. 12, 2019.",
 "M. Dayarathna, Y. Wen, and R. Fan, “Data Center Energy Consumption Modeling: A Survey,” IEEE Communications Surveys & Tutorials, vol. 18, no. 1, 2016.",
 "H. Kahil, S. Sharma, P. Välisuo, and M. Elmusrati, “Reinforcement learning for data center energy efficiency optimization: A systematic literature review and research roadmap,” Applied Energy, 2024.",
 "Y. Ran, H. Hu, X. Zhou, and Y. Wen, “DeepEE: Joint Optimization of Job Scheduling and Cooling Control for Data Center Energy Efficiency Using Deep Reinforcement Learning,” in Proc. IEEE ICDCS, 2019.",
 "X. Fan, W.-D. Weber, and L. A. Barroso, “Power provisioning for a warehouse-sized computer,” in Proc. ISCA, 2007.",
 "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017.",
 "V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, 2015.",
 "A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” JMLR, vol. 22, 2021.",
 "D. G. Feitelson, Workload Modeling for Computer Systems Performance Evaluation. Cambridge University Press, 2015.",
 "J. Wilkes, “Google cluster-usage traces v3,” Google Inc., technical documentation distributed with ClusterData2019, rev. 2020-08.",
 "A. Qureshi, R. Weber, H. Balakrishnan, J. Guttag, and B. Maggs, “Cutting the Electric Bill for Internet-Scale Systems,” in Proc. ACM SIGCOMM, 2009.",
 "L. Rao, X. Liu, L. Xie, and W. Liu, “Minimizing Electricity Cost: Optimization of Distributed Internet Data Centers in a Multi-Electricity-Market Environment,” in Proc. IEEE INFOCOM, 2010.",
 "Z. Liu, M. Lin, A. Wierman, S. H. Low, and L. L. H. Andrew, “Geographical Load Balancing with Renewables,” ACM SIGMETRICS Performance Evaluation Review, vol. 39, no. 3, 2011.",
 "A. Naug, A. Guillen, R. Luna, V. Gundecha, D. Rengarajan, S. Ghorbanpour, S. Mousavi, A. Ramesh Babu, D. Markovikj, L. D. Kashyap, and S. Sarkar, “SustainDC: Benchmarking for Sustainable Data Center Control,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.",
 "Hewlett Packard Enterprise, “SustainCluster: A high-fidelity, open-source Gymnasium environment for benchmarking multi-objective, sustainable workload scheduling across geo-distributed data centers,” software repository, https://github.com/HewlettPackard/sustain-cluster (MIT License).",
 "International Energy Agency, “Key Questions on Energy and AI,” World Energy Outlook Special Report, IEA, 2025.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    run = p.add_run(f"[{i}] {r}")
    run.font.size = Pt(8)
    p.paragraph_format.space_after = Pt(2)

doc.save(OUT)
print(f"Wrote {OUT}")
