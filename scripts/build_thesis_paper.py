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
P("Master's Thesis Draft — generated working paper; numbers correspond to "
  "repository state at commit lineage of 2026-06-10", italic=True, size=8,
  align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

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
  "(R² = 0.75–0.80 per cell vs. 0.43 pooled), real ISO prices, and EIA-930 "
  "net-demand series. A continuous-action PPO policy controlling routing fractions, "
  "batch drain rates, and batch placement defeats every baseline in all four "
  "evaluated configurations: 1.8–11.7% cheaper than the grid-unaware status "
  "quo, 2.7–9.7% cheaper than a foresighted lookahead oracle, and 1.0–8.3% "
  "cheaper than discrete DQN variants, with zero deadline violations and a "
  "~14 MW lower fleet peak. We additionally contribute three experimentally "
  "validated modeling lessons — deadline penalties must scale with the energy value "
  "of deferred work; capacity-blocked work must queue, not expire; and synthetic "
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
  "ClusterData 2019 [1], PowerData2019 [2], regional ISO prices, and EIA-930 net "
  "demand. (C2) Ground-truth measured per-tier demand decomposition (SLO service "
  "vs. no-SLO batch) of four trace cells, with the finding that resource requests "
  "overestimate the deferrable usage share by 5–13×. (C3) A continuous-"
  "action PPO formulation (routing + drain + batch placement) that wins every "
  "evaluated configuration against oracle, DQN, and status-quo baselines with "
  "zero deadline violations. (C4) Identification of marginal-power (slope) "
  "arbitrage as an emergent routing strategy enabled by per-cell power "
  "calibration. (C5) Three transferable modeling lessons for deferral-with-"
  "deadlines simulators, each validated by ranking inversion. (C6) An honest "
  "characterization of the temporal lever (+1.5–1.6% at measured deferrable "
  "fractions) and of the time-uniformity of the learned advantage.")

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
  "rather than carbon). SustainCluster's transmission-cost model is the "
  "natural template for the movement-cost sensitivity analysis of Sec. VIII.")

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
P("The agent outputs a ∈ [−1,1]³ᴺ, decoded as service routing "
  "fractions f = softmax(a₁:ₙ), drain rates δ = σ(aₙ₊₁:₂ₙ), "
  "and batch placement h = softmax(a₂ₙ₊₁:₃ₙ) — batch "
  "work, having no latency SLO, may execute anywhere:")
EQ("uᵢ,ₜ = min( fᵢ Dₜ + Bᵢ,ₜ + hᵢ Σⱼ δⱼ Qⱼ,ₜ ,  κᵢ )", 7)
P("with Dₜ total service demand, B backlog (unserved service, penalized), "
  "and service strictly prioritized over batch. The reward is")
EQ("rₜ = −[ Eₜ + Φₜ + λ_b Σ B + λ_κ Σ max(0, u−κ) + w Σ X ]", 8)
P("with λ_b = 1.5, λ_κ = 5, and deadline weight w = 250, calibrated "
  "to the energy cost of serving one unit of deferred work (Sec. VII, L1) so "
  "that discarding committed work is never rational. The demand-smoothing "
  "metric reported alongside cost is the fleet load factor LF = mean(g)/max(g).")

# ---------------- IV. DATA ----------------
H1("IV. DATA AND CALIBRATION")
H2("A. Cells as proxy data centers")
P("Cells a–d serve as four proxy DCs (US scenario: OR/IA/GA/SC on CAISO, "
  "MISO, Southern Co., Duke; Global scenario: OR/IA/NL/SG on CAISO, MISO, "
  "ENTSO-E, EMA). Tirmazi documents “considerable inter-cell workload "
  "variation” but no geography; the cell-as-DC mapping is a stated modeling "
  "exercise. The normalized curve provides the demand shape; R = 100 MW sets "
  "hyperscale magnitude (a real cell is ~3–5 MW, invisible to a regional "
  "grid). Sensitivity to R is future work.")
H2("B. Measured per-tier demand")
P("The service/batch split is measured, not modeled: instance_usage is joined "
  "to collection priority and aggregated per 5-minute bucket into service (SLO "
  "tiers) and batch (no-SLO tiers) curves with v + a = w to machine precision "
  "(Fig. 2). Measured deferrable usage shares are 2.1% (a), 6.1% (b), 8.1% (c), "
  "9.7% (d). Request-weighted proxies — job counts × requested CPU × "
  "duration — suggested 27–61%: an overestimate of 5–13×, "
  "consistent with Borg's over-allocation of best-effort tiers (cell c allocates "
  "~140% of capacity to best-effort batch alone [1]). Requests measure intent; "
  "usage measures schedulable reality.")
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

# ---------------- V. EXPERIMENTAL SETUP ----------------
H1("V. EXPERIMENTAL SETUP")
H2("A. Agents and baselines")
P("PPO (Stable-Baselines3, MLP 128×128, lr 3e-4, 500k steps) acts in the "
  "continuous 3N-dimensional space. Two DQN variants share hyperparameters "
  "(MLP 256×256, replay 100k, target update 1k): a 759-action routing grid, "
  "and a 48-action CFWS-style flattened index decoding to (source DC, "
  "destination DC, drain level) [6]. Nine heuristics span the policy space, "
  "including Round Robin, price-chasing and slack-grid concentrators, "
  "defer-to-trough rules, a GreenSlot-style Trough-Slot Lookahead with oracle "
  "3-hour net-demand foresight [10], and the no-optimization Status Quo "
  "(serve locally, immediately) — the counterfactual of a CICS-style layer "
  "switched off.")
H2("B. Configurations")
P("{US, Global} × {legacy = spatial only, batch = spatial + temporal}. "
  "All policies are evaluated deterministically on the identical full-trace "
  "episode (same seed).")
H2("C. Validation invariant")
P("By construction service + batch = measured demand, so a serve-everything-"
  "now policy must reproduce the legacy run exactly. This holds to the dollar: "
  "Status Quo scores $9,847,536 in both US modes with zero expiry — the batch "
  "machinery is demand-neutral, and any batch-mode difference between policies "
  "is scheduling, not artifact. This invariant caught two of the three modeling "
  "errors of Sec. VII.")

# ---------------- VI. RESULTS ----------------
H1("VI. RESULTS")
TABLE(["Config", "Status Quo", "Trough (oracle)", "Best DQN", "PPO", "Δ vs SQ"],
      [["US legacy", "9.848", "9.934", "9.759", "9.666", "−1.8%"],
       ["US batch", "9.848", "9.908", "9.798", "9.524", "−3.3%"],
       ["Glob legacy", "14.062", "13.751", "13.760", "12.617", "−10.3%"],
       ["Glob batch", "14.062", "13.755", "13.006", "12.421", "−11.7%"]],
      "TABLE I. TOTAL EPISODE COST (M$), FINAL ENVIRONMENT",
      widths=[0.85, 0.75, 0.85, 0.7, 0.6, 0.65])
FIG(FIGS / "fig1_results.png",
    "Fig. 1. Final results: PPO wins every configuration, with savings vs. the "
    "grid-unaware status quo annotated.")
H2("A. Headline")
P("PPO is the best policy — learned or heuristic — in all four configurations: "
  "1.8–11.7% below the status quo, 2.7–9.7% below the foresighted "
  "oracle, 1.0–8.3% below the best DQN variant per configuration, with "
  "zero deadline violations in batch mode and the fleet peak reduced from "
  "~303 to ~289 MW (Fig. 4). Savings scale with exploitable structure: "
  "modest under US-only diversity, large under global price/timezone spread.")
FIG(FIGS / "fig4_profile.png",
    "Fig. 4. Fleet power profile, US batch (first 4 days): PPO serves the same "
    "work at lower, flatter draw.")
H2("B. Emergent slope arbitrage")
P("Per-cell power calibration converted the low-diversity US scenario from "
  "“nothing to learn” (under pooled power, near-uniform routing is "
  "optimal) into a real optimization: idle power is sunk, so marginal load is "
  "cheapest where the slope is lowest. PPO inverts the load distribution "
  "relative to the do-nothing policies — toward low-slope cells a/b, away from "
  "high-slope c/d — worth +1.8% in US legacy. No baseline encodes this "
  "strategy. In the Global scenario, slope arbitrage compounds with price "
  "arbitrage: PPO cuts Singapore energy 32% and EU 21% vs. the status quo, "
  "repaying it with cheap low-slope US capacity.")
H2("C. The oracle loses everywhere")
P("Trough-Slot Lookahead holds privileged 3-hour future net-demand information "
  "yet loses every configuration (−2.7 to −9.7% vs. PPO) — and in the "
  "US loses even to Round Robin. Its slack-grid concentration raises the fleet "
  "peak (LF ≈ 0.86 vs. PPO's 0.96), and it is blind to per-cell power. "
  "Foresight does not compensate for optimizing the wrong surface; the learned "
  "policy internalizes the true cost structure.")
H2("D. The temporal lever, honestly sized")
P("At measured deferrable fractions (2–10%), batch deferral adds +1.5% "
  "(US) and +1.6% (Global) over PPO's own legacy result, with deadline "
  "violations essentially zero for every reasonable policy. Earlier inflated "
  "estimates (+6%) traced to artifacts L1–L3 (Sec. VII). Larger deferral "
  "leverage is a sensitivity question for batch-heavier mixes, addressable "
  "with the retained generator of Sec. II-B.")
H2("E. DQN fragility")
P("Across eight DQN runs, three failed distinctly: the routing grid diverged "
  "in Global legacy (a degenerate concentration policy: lowest energy, "
  "LF = 0.996, massive backlog penalties, $77.8M total) and degraded in US "
  "batch; the flat-index variant converged to a near-uniform no-op in Global "
  "batch (within $4 of Round Robin). Neither encoding dominates the other, "
  "and PPO trained reliably in all configurations with one hyperparameter "
  "set. In this formulation the continuous/discrete choice matters for both "
  "expressiveness (slope arbitrage requires fractional control) and training "
  "stability. These observations concern our cell-aggregate formulation, not "
  "CFWS's per-VM setting, where the encoding is reported effective [6].")
H2("F. Where the advantage lives in time")
P("Decomposing PPO's per-step advantage over the status quo by net-demand "
  "quartile (Fig. 5): savings are nearly uniform — the peak quartile earns "
  "only ~1.25× the slack quartile per step (US), and the Global profile "
  "is flat. The arbitrage operates continuously rather than in rare crisis "
  "windows. This inverts an earlier finding (2.1× burst-window "
  "concentration) measured on an environment whose batch bursts were later "
  "shown to be synthetic artifacts — the inversion itself evidences the "
  "correction (Sec. VII, L3).")
FIG(FIGS / "fig5_peakwindow.png",
    "Fig. 5. PPO savings by grid net-demand quartile: near-uniform in time, "
    "with a mild US tilt toward the duck-curve neck.")

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
  "(cost − w₀·x) + w·x; the ranking inverts at w ≈ 183. "
  "We set w = 250 ≈ the energy cost of one served unit.")
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
  "overestimated the deferrable fraction 5–13×. A one-line shape "
  "check (generated vs. source peak/mean) catches both.")
P("Threats to validity. (i) Cells are not geographically distributed; the "
  "cell-as-DC mapping is a modeling exercise justified by documented inter-cell "
  "heterogeneity [1]. (ii) The 100 MW magnitude is an assumption bridging "
  "cell scale to grid relevance; results should be swept over R. (iii) Single "
  "training seed per configuration; DQN's failures and PPO's margins warrant "
  "multi-seed confirmation. (iv) The measured tier curves make episodes "
  "deterministic; the agent may partially memorize the calendar. Held-out "
  "cells e–h and price/net-demand year shifts are the natural "
  "generalization tests. (v) Cooling and PUE are excluded by scope; a "
  "multiplicative PUE would scale, not reorder, results. (vi) The linear "
  "power model omits memory/IO effects (+0.03 R² at most in our data).")

# ---------------- VIII. FUTURE WORK ----------------
H1("VIII. FURTHER TOPICS OF CONSIDERATION")
P("(1) Sensitivity sweeps: deferrable fraction (via the retained generator) "
  "up to the ~20%-of-capacity trace-wide batch average [1]; rated power R; "
  "flexibility factor φ; peak weight α. (2) Generalization: held-out "
  "cells e–h, different trace months, shifted price/net-demand years, and "
  "multi-seed statistics. (3) Carbon objective: swap or add marginal "
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
P("On an environment assembled end-to-end from measured public data — per-tier "
  "demand from ClusterData 2019, per-cell power from PowerData2019, real "
  "prices and net demand — a continuous-action PPO agent learns grid-aware "
  "spatio-temporal load shaping that beats the grid-unaware status quo by "
  "1.8–11.7%, a foresighted oracle by 2.7–9.7%, and discrete DQN "
  "variants by 1.0–8.3% in every configuration, with zero deadline "
  "violations and a flatter, lower fleet power profile. The savings come from "
  "continuously exploitable structure — price spreads and per-cell marginal-"
  "power differences — rather than rare crisis windows. Equally important, "
  "the path to these numbers required finding and fixing three modeling "
  "errors that each inverted the ranking; we offer the corrected environment, "
  "the validation invariant, and the lessons themselves as contributions to "
  "reproducible research in sustainable computing.")

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
