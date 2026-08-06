# -*- coding: utf-8 -*-
"""Build thesis_paper.docx — a two-column, IEEE-style thesis paper at repo root.

DOCX is chosen for direct import into Google Docs. Math uses Unicode notation
(editable text, survives import). Current energy figures come from the v2 gate;
measured tier/power figures remain under output/paper_figs/.
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
ENERGY_FIGS = ROOT / "output" / "energy_model_v2" / "2025"
OOF_FIGS = ROOT / "output" / "oof_v2_2025"
OUT = ROOT / "thesis_paper.docx"

doc = Document()
EMIT_CONTENT = True
zoom = doc.settings.element.find(qn("w:zoom"))
if zoom is not None:
    zoom.set(qn("w:percent"), "100")

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
    if not EMIT_CONTENT:
        return None
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
    if not EMIT_CONTENT:
        return None
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(2)


def EQ(text: str, num: int):
    if not EMIT_CONTENT:
        return None
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text + "        (" + str(num) + ")")
    r.italic = True
    r.font.size = Pt(9.5)
    p.paragraph_format.space_after = Pt(4)


def FIG(path: Path, caption: str, *, width=3.2):
    if not EMIT_CONTENT:
        return None
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    r.font.size = Pt(8)
    c.paragraph_format.space_after = Pt(6)


def TABLE(header: list[str], rows: list[list[str]], caption: str, widths=None):
    if not EMIT_CONTENT:
        return None
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
P("Master's Thesis Draft — frozen energy-model v2 campaign: 80 PPO models, "
  "two symmetric held-out workload folds, and 10 optimizer seeds/configuration. "
  "The broad joint-shaping headline criterion failed; v1 remains historical. "
  "A post-hoc exploratory v3 recovery study (Sec. VII) partially repairs a v2 "
  "state-observability defect and re-screens PPO under a stricter safety-first gate; "
  "it still fails, reinforcing rather than reversing the frozen v2 conclusion.",
  italic=True, size=8, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

P("Abstract — Hyperscale data centers are now grid-scale electrical loads whose "
  "consumption coincides with regional net-demand peaks (the “duck curve”). "
  "We study whether a reinforcement-learning agent can route workload across "
  "geographically distributed data centers — spatially (where) and temporally "
  "(when, for deferrable batch work) — to minimize electricity cost while "
  "reducing the fleet's contribution to grid stress. The active experiment "
  "combines Google ClusterData 2019 workload shapes and PowerData2019-calibrated "
  "power models with one co-timestamped, real May-2025 CAISO archetype: native "
  "five-minute net demand and solar plus NP15 day-ahead LMP. Price and net "
  "demand are shifted together by local wall time across four US slots and "
  "four global slots, preserving negative prices and signed net demand over a "
  "complete 8,928-step (744-hour) experiment. Measured workload shapes map to "
  "equal 100 MW/unit-capacity proxies, and current batch arrivals are observed "
  "before same-step release. Routing is unrestricted and therefore represents "
  "an optimistic upper bound, not a deployment claim. The corrected "
  "clairvoyant-QP gate finds 8.0–17.4% joint headroom, of which only 0.6–2.3% "
  "is incremental temporal value. We then train 80 PPO models under a frozen "
  "two-fold a–d/e–h protocol with 10 seeds/configuration and no test-fold "
  "selection. Global spatial PPO is the only robust learned success, saving "
  "0.90% and 1.19% on the two held-out folds; both optimizer-bootstrap CIs and "
  "wider t-interval sensitivities remain positive. US spatial savings are not "
  "established (−0.29% and +0.07%). Joint batch control fails the frozen "
  "headline: only 9/40 batch seeds meet the 99.99% completion floor, and joint "
  "PPO does not reliably improve over separately trained spatial PPO. Spatial "
  "PPO lowers the secondary demand-charge reference, whereas joint PPO raises "
  "it and worsens rare maximum three-hour ramps. Thus the evidence supports a "
  "small optimistic Global spatial effect, not reliable joint spatio-temporal "
  "optimization. A post-hoc exploratory v3 study adds episode progress and "
  "deadline buckets and re-screens PPO across a GAE/KL/learning-"
  "rate/idle-subtraction/potential/normalization sweep and a three-point budget "
  "curve (151,552–1,003,520 steps) under a stricter safety-first gate requiring "
  "all ten seeds feasible and a positive optimizer-bootstrap interval. It still "
  "fails: the safety-first-selected US configuration is 1/10 feasible with a "
  "CI crossing zero, and the Global configuration is 1/10 feasible despite a "
  "positive CI, so unconstrained PPO remains unsuitable as a trustworthy joint "
  "controller under this model.", size=9)
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
  "System (CICS) demonstrates temporal shifting at hyperscale [3]; geographic "
  "load-balancing work establishes the spatial lever [23]–[25]. Open "
  "RL environments for sustainable data-center scheduling now exist — "
  "SustainDC for within-DC multi-agent control [26] and SustainCluster for "
  "geo-distributed per-task dispatch [27] — but none provides measured "
  "per-tier (flexible vs. inflexible) demand decomposition of a hyperscale "
  "trace, power models calibrated on companion power measurements, or a grid "
  "net-demand (duck-curve) objective; those are the gaps this thesis fills.")
P("This thesis asks: can a model-free reinforcement-learning agent, observing "
  "only per-site demand, prices, and grid net demand, learn a joint spatial-"
  "temporal load-shaping policy that beats the grid-unaware status quo? The "
  "frozen evidence gives a qualified answer: Global spatial routing transfers "
  "with a small positive effect, US spatial savings are not established, and "
  "joint batch control is unstable and fails the completion criterion. The "
  "energy/system construction succeeds as a reproducible test bed; the broad "
  "learned joint-optimization claim does not.")
P("Contributions. (C1) A reproducible Gymnasium environment for multi-data-center "
  "grid-aware load shaping at cell-aggregate granularity, grounded in Google "
  "ClusterData 2019 [1] and PowerData2019 [2]. (C2) Ground-truth measured per-tier demand "
  "decomposition (SLO service vs. no-SLO batch) of four trace cells, with the "
  "finding that resource requests overestimate the deferrable usage share by "
  "3–5×. (C3) A real, reproducible May-2025 CAISO duck-curve/price archetype "
  "whose paired signals are shifted by IANA local wall time across US and global "
  "slots. (C4) A no-training QP gate that sizes the opportunity before policy "
  "selection and shows that spatial diversity dominates temporal flexibility. "
  "(C5) A provenance-locked 80-model held-out campaign showing a robust but "
  "small Global spatial effect and a negative result for reliable joint batch "
  "control. (C6) A preserved audit trail of the retired mixed/synthetic model "
  "and the simulator corrections that prevented invalid results from being "
  "promoted. (C7) A post-hoc exploratory v3 state-repair and budget-scaling "
  "study (Sec. VII) showing the v2 joint failure survives an augmented state "
  "representation and a 2× larger training budget, so it is not merely an "
  "observability artifact.")

# ---------------- II. RELATED WORK ----------------
H1("II. RELATED WORK AND METHODOLOGICAL LINEAGE")
P("The current thesis combines four established ideas rather than claiming a "
  "new scheduling paradigm: measured aggregate workload shaping, geographic "
  "load balancing, deadline-constrained deferral, and model-free control. This "
  "section identifies only the lineage that remains active in energy-model v2.")
H2("A. Aggregate workload shaping and trace semantics")
P("ClusterData 2019 [1] defines a Borg cell as one management unit and reports "
  "five-minute, normalized resource usage across eight heterogeneous cells. We "
  "use those measured cell aggregates directly. Collection priority separates "
  "the no-SLO free/best-effort tiers (priority ≤115 [22]) from SLO-bearing "
  "service; no synthetic arrival stream is used in the primary experiment. "
  "PowerData2019 [2] supplies the companion per-cell power calibration.")
P("Google's CICS [3] is the main production precedent for this granularity. It "
  "shapes aggregate flexible and inflexible cluster demand independently of the "
  "real-time job scheduler, using cluster-specific forecasts and power models. "
  "Our temporal abstraction follows that aggregate view, while the objective "
  "changes from carbon to electricity cost and grid net demand. CICS is a "
  "precedent, not a reproduced system, and it does not supply our spatial head.")
H2("B. Geographic load balancing for electricity cost")
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
H2("C. Deadline-constrained deferral")
P("GreenSlot [10] and related single-DC work establish the basic temporal idea: "
  "delay flexible work toward predicted cheap or renewable-rich intervals. "
  "Grange et al. [4] add due-date constraints; Xu et al. [7] distinguish "
  "interactive from deferrable work; Haghshenas et al. [8] and Liu et al. [9] "
  "combine heterogeneous load with prices or forecasts. We borrow the pool-and-"
  "deadline abstraction, not their job-level placement mechanisms. In our "
  "aggregate model, measured no-SLO demand enters a queue and the controller "
  "selects its release rate and execution site.")
H2("D. Reinforcement learning and open environments")
P("Model-free RL is established for data-center control [15], with DeepEE [16] "
  "as one example of joint scheduling and infrastructure control. We use PPO "
  "[18] because the current action is continuous: routing fractions, release "
  "rates, and batch-placement fractions. The paper does not claim a new RL "
  "algorithm; its contribution is the measured workload/energy formulation and "
  "the controlled evaluation around it.")
P("SustainDC [26] addresses multi-agent control within a data center, while "
  "SustainCluster [27] dispatches individual Alibaba GPU tasks across many "
  "locations with carbon, cost, SLA, and transmission objectives. They are "
  "complementary rather than drop-in baselines: this thesis controls divisible "
  "aggregate Google-cell demand and explicitly targets CAISO net-demand shape. "
  "Reproducing the same question in either benchmark would require changing its "
  "workload granularity or objective.")
H2("E. Current positioning")
P("The active contribution is a controlled geo-distributed workload-shaping "
  "experiment that combines measured aggregate service/batch curves, per-cell "
  "power models, and one real May-2025 CAISO net-demand/price archetype shifted "
  "across US and global local-time slots. A clairvoyant QP sizes spatial and "
  "temporal headroom before PPO is trained. This is not a real multi-market "
  "replay, a CICS reproduction, or a VM/job scheduler, and no archived v1 "
  "learned-policy result is promoted as v2 evidence.")

# ---------------- III. SYSTEM MODEL ----------------
H1("III. SYSTEM MODEL")
P("N = 4 data centers, indexed by i, j ∈ {1,…,N} (i is the site under "
  "consideration; j is a running index used in fleet-wide sums), operate over "
  "T = 8,928 five-minute steps, indexed by t, τ ∈ {1,…,T} (t is the current "
  "step; τ is a running index over earlier steps), with Δ = 300 s and "
  "Δh = 1/12 h. Site i has measured aggregate demand curve wᵢ,ₜ ∈ [0,1] "
  "(fraction of cell capacity), decomposed by measured priority tier into "
  "non-deferrable service demand vᵢ,ₜ and deferrable batch arrivals aᵢ,ₜ with "
  "wᵢ,ₜ = vᵢ,ₜ + aᵢ,ₜ exactly. The total service demand pooled across the "
  "fleet at step t is Dₜ = Σᵢ vᵢ,ₜ — the quantity the routing fractions "
  "redistribute. Site inputs further include electricity price πᵢ,ₜ ($/kWh), "
  "signed grid net demand dᵢ,ₜ ∈ [−1,1] (negative = renewable oversupply, "
  "positive = residual grid load), and normalized proxy capacity κᵢ = 1.")
H2("A. Equal-capacity proxy, power, and cost")
P("Each wᵢ,ₜ is already measured CPU usage divided by its SOURCE cell's CPU "
  "capacity. We use that real curve as a utilization shape and map it onto an "
  "equal 100 MW proxy DC. Therefore every destination has κᵢ = 1: w = 0.70 "
  "means 70% of that proxy's capacity. Raw machine totals remain provenance "
  "metadata and support calibration; they do not shrink destination capacity "
  "again. This avoids mixing own-cell utilization units with capacity expressed "
  "relative to the largest source fleet.")
P("Each site has a per-cell calibrated linear power model — the standard "
  "idle+slope server model [14], [17] (Sec. IV-C):")
EQ("Pᵢ(u) = Pᵢⁱᵈˡᵉ + sᵢ · u", 1)
EQ("gᵢ,ₜ = Pᵢ(uᵢ,ₜ) · R,   R = 100 MW", 2)
P("where uᵢ,ₜ∈[0,1] is served utilization of the equal 100 MW proxy, "
  "Pᵢⁱᵈˡᵉ is cell i's idle power (its draw at zero "
  "CPU, as a fraction of theoretical peak), sᵢ is its marginal power slope "
  "(the extra power per unit of CPU and an observable routing context), R is "
  "rated power per site, and g is the grid draw "
  "(sites are pure grid loads; no on-site generation). Energy cost and the "
  "quadratic, net-demand-weighted peak-contribution penalty are")
EQ("Eₜ = Σᵢ πᵢ,ₜ · gᵢ,ₜ · 1000 · Δh", 3)
EQ("Φₜ = α · Σᵢ gᵢ,ₜ² · max(dᵢ,ₜ,0)", 4)
P("The quadratic form penalizes concentration; the d-weighting makes draw "
  "near the duck-curve neck expensive. Signed d remains observable, while "
  "max(d,0) keeps the QP convex. At negative net demand Φ is zero; the real "
  "low or negative LMP provides the economic reward for moving work into the "
  "duck-curve belly. Φ is a grid-stress SHADOW PRICE, not a tariff: α must be "
  "selected by a documented v2 component/sensitivity study, not inherited from "
  "v1 or derived from a rate schedule, and Φ "
  "differs from a commercial demand charge in all three respects that matter — "
  "it is summed over every interval rather than taken as a maximum, it is "
  "quadratic rather than linear in kW, and it is weighted by grid net demand, "
  "which no tariff observes.")
P("All slots use the same CAISO archetype and signed MW denominator, so d is "
  "comparable by construction inside this controlled experiment. This is not "
  "a claim that real California, Amsterdam, and Singapore systems have equal "
  "physical scarcity.")
P("Ramp-rate treatment — evaluated, not optimized.", bold=True)
P("The duck curve combines a deep midday trough with a steep upward evening "
  "ramp. The primary objective addresses trough economics and HIGH net-demand "
  "EXPOSURE, but contains no derivative dₜ−dₜ₋H; it does not directly optimize "
  "ramp rate. Price is an incomplete proxy: its correlation is 0.895 with "
  "net-demand level but only 0.206 with the one-hour ramp and 0.429 with the "
  "three-hour ramp.")
P("This omission is deliberate for the first v2 campaign. Concentrating work "
  "during negative-demand, very-low-price intervals is acceptable within proxy "
  "capacity, and Φ is zero there. Evaluation reports per-region maximum and "
  "95th-percentile upward ramps for H=1 hour and H=3 hours, comparing "
  "max(0,Nₜ−Nₜ₋H) with max(0,(Nₜ+gₜ)−(Nₜ₋H+gₜ₋H)). Regions are not summed into "
  "a fictitious global grid. A ramp-aware reward is future work and will be "
  "introduced only if the first v2 policies worsen these physical KPIs. Status "
  "Quo changes the maximum one-hour regional ramp by only −0.4 to +1.7 MW "
  "across scenarios against an 11,465 MW raw CAISO maximum; trained-policy "
  "effects remain unknown.")
P("Because that distinction is easy to lose, the operator's actual tariff term "
  "is modeled separately. Commercial and industrial customers are billed on the "
  "single highest demand interval of each billing period, per meter. Such charges "
  "can be a substantial C&I bill component, but the share is tariff- and "
  "customer-specific; our reference sensitivity assumes no universal percentage. "
  "The billed quantity is the sum of per-SITE maxima, not the fleet coincident peak:")
EQ("Ψ = c · Σᵢ maxₜ gᵢ,ₜ · 1000,   c in $/kW per billing period", 5)
P("A maximum over the period is not a per-step cost, so Ψ is charged in the "
  "telescoping form below, where Dᵢ,ₜ = maxₜ′≤ₜ gᵢ,ₜ′ is the running billed peak. "
  "Each step pays exactly the amount by which it raises the running maximum, "
  "and the sum over a period is exactly Ψ:")
EQ("ψᵢ,ₜ = c · max(0, gᵢ,ₜ − Dᵢ,ₜ₋₁) · 1000,   Σₜ ψᵢ,ₜ = c · maxₜ gᵢ,ₜ · 1000", 6)
P("The complete 8,928-step episode is one study billing cycle by default, so c "
  "is applied exactly once and the in-reward quantity matches the post-hoc "
  "metric. A shorter configured period is a different tariff with its own "
  "$/kW-period rate, not a monthly proxy. The telescoping identity is exact "
  "under the RL objective only when gamma = 1; the training entry points "
  "therefore select gamma = 1 whenever the term is enabled and reject a lower "
  "discount. The observation then includes Dᵢ,ₜ₋₁ for every site plus billing-"
  "period progress and the active-rate fraction, and a new period is reset "
  "before its first observation. Together these make the tariff state Markov "
  "and prevent actions from seeing a stale prior-period peak. The v2 primary "
  "protocol fixes c=0 in the reward and reports the standardized $15/kW-cycle "
  "charge as a secondary sensitivity for every policy. Demand-aware training "
  "is a separate extension.")
P("The $15/kW reference is a sensitivity analysis rather than a reconstruction "
  "of four utility bills. It is a mid-range US commercial/industrial reference "
  "within the broad tariff range catalogued by the NREL demand-charge survey "
  "[29]; it uses that one US rate across all geographies solely for comparison, treats each "
  "five-minute trace sample as a demand interval, aligns the cycle to the "
  "episode, and omits site-specific time-of-use demand tiers, ratchets, "
  "contract demand, and power-factor clauses.")
P("The reference charge is about $4.694M for Status Quo, large enough to "
  "dominate the controlled objective, and it is not a real Dutch or Singapore "
  "tariff. These are the reasons it is not primary.")
H2("B. Deferrable batch dynamics")
P("Each origin site maintains a batch pool with per-entry deadlines. Queue "
  "timing must be explicit: Q⁻ᵢ,ₜ is work carried into step t before the current "
  "arrival and deadline boundary; Q⁺ᵢ,ₜ is the work eligible for release after "
  "arrival aᵢ,ₜ is added and expired work Xᵢ,ₜ is removed. The per-cell deadline "
  "horizon follows fitted mean duration μᵢ and flexibility factor φ = 1 (after "
  "[4]). An arrival at t is eligible during steps t,…,t+Hᵢ−1 and expires at the "
  "start of t+Hᵢ if unfinished:")
P("Because the current action may release part of aᵢ,ₜ in the same step, "
  "aᵢ,ₜ is included explicitly in the pre-action observation alongside Q⁻, "
  "service demand, urgency, price, net demand, and site context. The decision "
  "process therefore observes every state variable that determines its "
  "same-step transition.")
P("The trace contains no job deadlines. Hᵢ is therefore an experimental control "
  "semantic, not a recovered SLO: μᵢ is fitted from observed duration and φ "
  "sets counterfactual slack. The primary model freezes φ=1 (H=2μ); robustness "
  "uses φ∈{0,2}, corresponding to H∈{μ,3μ}.")
EQ("Hᵢ = ⌈ μᵢ (1 + φ) / Δ ⌉", 7)
EQ("Q⁺ᵢ,ₜ = Q⁻ᵢ,ₜ + aᵢ,ₜ − Xᵢ,ₜ;   rᵢ,ₜ = δᵢ,ₜQ⁺ᵢ,ₜ;   Q⁻ᵢ,ₜ₊₁ = Q⁺ᵢ,ₜ − qᵢ,ₜ", 8)
P("Here rᵢ,ₜ is origin i's requested release and qᵢ,ₜ is the amount of that "
  "origin's work actually completed anywhere in the fleet. This two-phase "
  "notation resolves an indexing ambiguity in earlier drafts: if Q were instead "
  "defined as the post-arrival pool, its next-step recurrence would indeed use "
  "aᵢ,ₜ₊₁. With Q⁻ defined before the arrival, aᵢ,ₜ in Eq. (8) is correct. A "
  "release is only an intent to run; capacity-blocked work remains in Q with "
  "its original deadline. Only completed q leaves the pool and only work at its "
  "deadline boundary becomes X.")
H2("C. Action, serving, and reward")
P("The agent outputs a ∈ [−3,3]³ᴺ, decoded as service routing "
  "fractions f = softmax(a₁:ₙ), drain rates δ = sigmoid(aₙ₊₁:₂ₙ), and batch "
  "placement h = softmax(a₂ₙ₊₁:₃ₙ) — batch work, having no latency SLO, may "
  "execute anywhere. Indices now distinguish origin i from execution site j. "
  "Service has first claim on each unit-capacity proxy; batch runs in the "
  "remainder:")
P("The primary controlled experiment assumes both service and batch are freely "
  "routable among all four slots, including intercontinental Global routing. "
  "Latency, data residency, and movement cost are not modeled. This is an "
  "explicit optimistic assumption, not an operational deployment claim.")
EQ("σⱼ,ₜ=min(fⱼ,ₜDₜ+Bⱼ,ₜ₋₁,κⱼ);  yⱼ,ₜ=min(hⱼ,ₜΣᵢrᵢ,ₜ,κⱼ−σⱼ,ₜ);  ρₜ=Σⱼyⱼ,ₜ/Σᵢrᵢ,ₜ;  qᵢ,ₜ=ρₜrᵢ,ₜ;  uⱼ,ₜ=σⱼ,ₜ+yⱼ,ₜ", 9)
P("In Eq. (9), σ is service executed at destination j, y is batch executed "
  "there, and q is completed work removed from origin i's queue. The global "
  "serve ratio ρ allocates any capacity shortfall proportionally across origin "
  "release requests (ρ = 0 when no work is requested), so Σᵢqᵢ,ₜ = Σⱼyⱼ,ₜ. "
  "Within each origin, completed work is removed earliest-deadline-first. This "
  "origin/destination distinction is required once batch placement is spatial; "
  "earlier notation incorrectly used one Sᵢ,ₜ for both roles. The ±3 action "
  "bound (not the SB3 default ±1) matters: SB3 clips actions to the action box "
  "before the decode, so ±1 would cap every routing share to [4.3%, 71%] and "
  "every drain rate to [27%, 73%], making full concentration and multi-hour "
  "holding impossible by construction; ±3 restores parity with the discrete "
  "baselines (shares to ~98.5%). The reward is the negative per-step cost:")
EQ("rₜ = −[ Eₜ + Φₜ + Σᵢ(ψᵢ,ₜ + λ_bBᵢ,ₜ) + χₜ ]", 10)
P("With the tariff disabled, backlog weight λ_b = 25 and expiry weight λ_x = "
  "250 preserve the historical objective. Enabling a demand charge also enables "
  "an economic guard: both weights are raised above a conservative bound on the "
  "largest one-step energy + grid-penalty + demand-charge saving obtainable by "
  "dropping one normalized CPU unit (with a 5% margin). This prevents the tariff "
  "from making unserved work economically optimal. Default-off χₜ=λ_xΣᵢXᵢ,ₜ. "
  "Demand-enabled runs use the exact dense potential "
  "χₜ=λ_xΣᵢ(aᵢ,ₜ−qᵢ,ₜ); with γ=1, Σₜχₜ=λ_x(expired+terminal-pool work). This "
  "credits completion immediately while preserving the finite-horizon objective. "
  "Raw dollar costs remain in logs and reports, while the RL reward is multiplied "
  "by 10⁻⁴ for numerical conditioning. Capacity is a "
  "hard serving constraint in Eq. (9), so no capacity-violation term belongs in "
  "the mathematical objective (the implementation retains a zero-valued "
  "diagnostic guard). The zero-expiry audit is an observed result, not a hard "
  "constraint. The demand-smoothing metric reported alongside "
  "cost is fleet load factor LF = mean(g)/max(g); independent physical ramp "
  "KPIs are reported separately and are not reward terms.")
H2("D. Worked three-step example")
P("A one-site slice makes the units concrete. Let κ=1, R=100 MW, "
  "P(u)=0.50+0.40u, α=0.015, c=$15/kW-cycle, and assume no backlog or expiry. "
  "The values below use Eq. (8)–(10); a multi-site step performs the same "
  "arithmetic after f and h divide service and batch across destinations.")
TABLE(
    ["Quantity", "t=0", "t=1", "t=2"],
    [
        ["D; Q⁻; a", "0.50; 0.10; 0.20", "0.40; 0.15; 0.10", "0.70; 0.15; 0"],
        ["δ; Q⁺; q=y", "0.50; 0.30; 0.15", "0.40; 0.25; 0.10", "1.00; 0.15; 0.15"],
        ["u=σ+y", "0.65", "0.50", "0.85"],
        ["g=(.50+.40u)100", "76 MW", "70 MW", "84 MW"],
        ["π; d", "$0.06; 0.80", "$0.04; 0.50", "$0.08; 0.90"],
        ["Energy E", "$380.00", "$233.33", "$560.00"],
        ["Grid penalty Φ", "$69.31", "$36.75", "$95.26"],
        ["Demand increment ψ", "$1.140M", "$0", "$0.120M"],
        ["Next Q⁻", "0.15", "0.15", "0"],
    ],
    "ILLUSTRATIVE THREE-STEP EQUATION WALKTHROUGH",
    widths=[0.95, 0.72, 0.72, 0.72],
)
P("For example, at t=2 the queue closes as Q⁻₃=0.15+0−0−0.15=0. "
  "Grid draw rises from the prior 76 MW record to 84 MW, so the demand charge "
  "adds 15×(84−76)×1000=$120,000. Energy and grid penalty still accrue every "
  "step, whereas ψ is zero whenever the running peak does not increase.")
H2("E. Problem formulation")
P("Putting the terms together, the agent solves a constrained cost-"
  "minimization over the full episode. With the per-step controls fₜ (service-"
  "routing fractions), δₜ (drain rates), and hₜ (batch placement) of Sec. III-C:")
EQ("minimize  J = Σₜ [ Eₜ + Φₜ + Σᵢ(ψᵢ,ₜ + λ_bBᵢ,ₜ) + χₜ ]", 11)
P("J contains no direct ramp-rate penalty. Any ramp improvement is therefore "
  "an independently measured outcome rather than a consequence built into the "
  "training objective.")
P("subject to, for all i,j,t: the power model (1)–(2); the serving and origin-"
  "completion rules (9) with uⱼ,ₜ≤κⱼ; routing and placement simplexes "
  "Σⱼfⱼ,ₜ=Σⱼhⱼ,ₜ=1 with f,h≥0 and δ∈[0,1]; service conservation "
  "Bⱼ,ₜ=Bⱼ,ₜ₋₁+fⱼ,ₜDₜ−σⱼ,ₜ; and the two-phase batch conservation (8). At each "
  "deadline boundary, every due arrival is either represented in cumulative "
  "origin completion q or charged once as expiry X. Thus deadlines are soft "
  "penalties in the environment, not hard constraints; the reported policies "
  "happen to achieve X=0. In words: route service and schedule batch to minimize "
  "energy, grid contribution, the actual demand tariff, backlog, and expiry "
  "cost while respecting capacity.")
P("Eq. (11) is an offline, full-information statement; the deployed controller "
  "is causal (it cannot observe future prices or demand). We therefore solve it "
  "with model-free RL. PPO maximizes 𝔼[Σₜγᵗrₜ]; when γ<1 this is not identical "
  "to undiscounted J, so older default-off results are evaluated on J but were "
  "trained with γ=0.99. Demand-charge-enabled runs set γ=1, aligning training "
  "and evaluation. The clairvoyant convex QP uses the same backlog and soft-"
  "expiry accounting, with fluid actions and perfect foresight; these relaxations "
  "make its optimum a lower bound on the evaluated J (Sec. IV-D).")
P("The formulation follows CICS's aggregate flexible/inflexible load-shaping "
  "problem [3], retargeted from carbon to grid net-demand and electricity cost, "
  "with the cost-minimization-over-distributed-DCs structure of geographic load "
  "balancing [23]–[25], a linear idle+slope power model [14], [17], a "
  "peak-contribution term in the peak-shaving and demand-response tradition "
  "[9], [13] — though Φ is a per-interval quadratic shadow price, not a demand "
  "charge in the tariff sense; the tariff term Ψ is modeled and reported "
  "separately (Eq. 5) — and aggregate batch-with-deadline "
  "dynamics [4], [9].")

# ---------------- IV. DATA ----------------
H1("IV. DATA AND CALIBRATION")
H2("A. Cells as proxy data centers")
P("Cells a–d serve as four proxy DCs. Cell locations in the trace are "
  "anonymized; assigning them to market slots is a modeling choice. Energy-"
  "model v2 uses Pacific/Mountain/Central/Eastern slots for the US experiment "
  "and Pacific/Central/Amsterdam/Singapore slots for the Global experiment, all "
  "driven by the same shifted CAISO archetype. Tirmazi documents considerable "
  "inter-cell workload variation but no geography. The normalized curve "
  "provides the own-cell utilization shape; κ = 1 maps every shape to an equal "
  "proxy destination; and R = 100 MW sets hyperscale magnitude (a real cell is "
  "~3–5 MW, invisible to a regional grid). Raw machine totals are metadata, not "
  "a second capacity normalization. Sensitivity to R is future work.")
H2("B. Measured per-tier demand")
P("The primary demand path starts in instance_usage, not the job table and not "
  "a generator. For each cell, we join collection priority, keep top-level "
  "instances with at least a five-minute usage record, group average CPU usage "
  "into 300-second buckets, and divide by measured cell CPU capacity. Priority "
  "≤115 is no-SLO batch; all higher priorities are service. We then define "
  "w = normalized total usage, a = normalized batch usage, and v = w − a. "
  "Consequently v + a = w at every interval to machine precision (Fig. 1).")
P("Each cell produces 8,929 ordinal workload rows. The v2 energy calendar has "
  "8,928 rows, so the loader removes the one extra workload boundary row and "
  "uses a complete 744-hour control episode. The source's discarded absolute "
  "timestamp is not reconstructed: workload timestep zero is explicitly "
  "anchored to May 1, 2025 00:00 PDT as a cross-year counterfactual.")
P("The source curves already equal measured usage divided by each cell's own "
  "CPU capacity. Because the modeled sites are equal 100 MW proxies, the loader "
  "sets destination capacity to 1.0 for every cell. Thus a source value of 0.70 "
  "remains 70% utilization after mapping; machine-count differences do not "
  "double-normalize it.")
TABLE(
    ["Cell", "Batch share", "Cell", "Batch share"],
    [
        ["a", "16.6%", "e", "23.5%"],
        ["b", "15.7%", "f", "8.4%"],
        ["c", "22.3%", "g", "37.7%"],
        ["d", "26.4%", "h", "29.1%"],
    ],
    "MEASURED NO-SLO SHARE OF AGGREGATE CPU USAGE",
    widths=[0.55, 0.95, 0.55, 0.95],
)
P("At runtime, v(t) is the must-serve input and a(t) is the measured arrival "
  "placed into its origin queue. The four-site PPO action routes pooled service, "
  "chooses each origin's release fraction, and places released batch across "
  "destinations. Service consumes capacity first; only completed batch is "
  "removed earliest-deadline-first, while capacity-blocked work keeps its "
  "original deadline. The full-month batch_fraction reported to the policy is "
  "context only; it never replaces the measured time-varying a(t) curve.")
P("The fitted job-distribution JSON supplies mean duration for the experimental "
  "deadline (and a memory ratio when memory modeling is enabled), but the "
  "BatchArrivalGenerator is not constructed when tier_curves is present. It "
  "therefore has no role in primary v2 training or scoring. Its only remaining "
  "purpose is optional counterfactual sensitivity to batch volume or arrival "
  "shape; it does not validate the measured trace.")
P("The frozen v2 evaluation will be symmetric and out of fold: train on a–d and "
  "score deterministic policies on e–h, then train on e–h and score on a–d, "
  "with ten seeds per fold/configuration. Training-only randomization may "
  "permute compute bundles and interpolate measured power parameters using only "
  "the training fold. Scoring uses the untouched held-out curves, the same v2 "
  "energy slots, Status Quo/Round Robin/Drain Immediately baselines, and the "
  "clairvoyant QP diagnostic. Reports include objective components, service "
  "completion, batch completion, expiry, and terminal backlog/pool so savings "
  "cannot be obtained by dropping work.")
FIG(FIGS / "fig2_tiers.png",
    "Fig. 1. Measured per-tier decomposition of cell b (3 of 31 days): "
    "service + batch equals the measured aggregate at every step.")
H2("C. Per-cell power calibration")
P("We adopt the standard linear idle+slope server model (Eq. 1) — the form "
  "used throughout the data-center power literature [14], [17] — and fix its "
  "two constants per cell from data rather than from a single textbook value. "
  "Joining hourly PowerData2019 measured power (whose ‘power utilization’ is "
  "defined exactly as our normalization: actual power over theoretical peak [2]) "
  "with aggregate CPU over 2,980 (cell, hour) samples, least-squares gives the "
  "per-cell idle and slope of Table II. The calibration is literature-"
  "consistent: the fitted idle draw is 0.40–0.61 of each cell's peak, consistent "
  "with the substantial idle fractions reported in server-power measurements "
  "and surveys [14], [17], so "
  "the constants are credible without appeal to goodness-of-fit. We keep PER-"
  "CELL constants (not one pooled value) because the cells span a meaningful "
  "proportionality range — cell d (idle 0.38, slope 0.57) is far more energy-"
  "proportional than cell a (0.53, 0.34) — and that BETWEEN-cell spread is "
  "a routing signal available to the agent. Preferring lower marginal slope "
  "when idle power is sunk is a direct consequence of this model, not a novel "
  "RL mechanism; "
  "a single literature constant would erase it. This mirrors CICS's choice of "
  "per-cluster power models [3]. (R² is reported in Table II as a diagnostic "
  "only — 0.43 pooled vs 0.75–0.80 per cell, confirming the pooled residual is "
  "between-cell heterogeneity rather than noise — but it is not the model's "
  "justification. Using one calibrated model gives a controlled comparison, but "
  "errors in marginal slopes could still change routing rankings; the "
  "held-out-cell and domain-randomization tests probe that sensitivity.)")
TABLE(["Cell", "idle", "slope", "R²"],
      [["a", "0.528", "0.340", "0.79"],
       ["b", "0.548", "0.376", "0.75"],
       ["c", "0.440", "0.526", "0.76"],
       ["d", "0.382", "0.570", "0.80"],
       ["pooled", "0.479", "0.444", "0.43"]],
      "TABLE II. POWER MODEL CALIBRATION (POWERDATA2019)",
      widths=[0.6, 0.7, 0.7, 0.6])
FIG(FIGS / "fig3_power.png",
    "Fig. 2. Per-cell power calibration: four distinct idle/slope lines; the "
    "pooled fit (dashed) blurs them into R² = 0.43.")
energy_gate = doc.add_section(WD_SECTION.CONTINUOUS)
set_cols(energy_gate, 1)
H2("D. Energy Model v2 Review Gate — May 2025")
P("No PPO retraining has been launched. This section validates the energy "
  "system and theoretical headroom first.", bold=True)
H2("Experimental energy model")
P("The primary model is one co-timestamped, real CAISO archetype: CAISO Today's "
  "Outlook native five-minute net demand; CAISO OASIS NP15 hourly day-ahead "
  "total LMP expanded stepwise to five minutes; negative prices preserved; one "
  "Pacific civil-time calendar with IANA/DST conversion; price and net demand "
  "shifted together across market slots; and no regional price re-averaging.")
P("US slots are Pacific, Mountain, Central, and Eastern. Global slots are "
  "Pacific, Central, Amsterdam, and Singapore. Google's May-2019 workload "
  "shapes are anchored to the May-2025 energy calendar as an explicit cross-"
  "year counterfactual.")
H2("Finite-window boundary handling")
P("The time-zone transformation is continuous rather than circular. Every slot "
  "contains exactly 8,928 five-minute intervals (744 hours), but shifted slots "
  "can use adjacent real CAISO hours at the month boundary instead of wrapping "
  "May 31 back to May 1. Singapore is 15 hours ahead of Pacific time, so its "
  "reference window replaces CAISO's first 15 hours of May (mean $21.54/MWh) "
  "with the first 15 hours of June (mean $29.16/MWh). Its monthly mean is "
  "therefore $26.09/MWh instead of Pacific's $25.93/MWh: a $0.154/MWh (0.59%) "
  "boundary effect, not extra simulated time or a Singapore price premium.")
P("The continuous shift is retained because a circular within-May shift would "
  "create an artificial May 31-to-May 1 discontinuity. Baselines, QP, and "
  "future learned policies are compared on the same slot data within each "
  "scenario; regional monthly means are not forced to match.")
H2("Reference diagnostics")
TABLE(
    ["Slot", "Mean", "Std", "r(price, net)", "Price peak UTC", "Net peak UTC"],
    [
        ["US Pacific", "$25.93", "$16.48", "0.895", "03:00", "03:00"],
        ["US Mountain", "$25.93", "$16.48", "0.895", "02:00", "02:00"],
        ["US Central", "$25.94", "$16.48", "0.895", "01:00", "01:00"],
        ["US Eastern", "$25.94", "$16.48", "0.895", "00:00", "00:00"],
        ["Global Pacific", "$25.93", "$16.48", "0.895", "03:00", "03:00"],
        ["Global Central", "$25.94", "$16.48", "0.895", "01:00", "01:00"],
        ["Global Amsterdam", "$25.94", "$16.47", "0.894", "18:00", "18:00"],
        ["Global Singapore", "$26.09", "$16.39", "0.892", "12:00", "12:00"],
    ],
    "ENERGY MODEL V2 REFERENCE DIAGNOSTICS (PRICE IN USD/MWh)",
    widths=[1.15, 0.75, 0.75, 0.85, 1.05, 1.05],
)
P("Reference CAISO facts: mean DAM price is about $25.93/MWh; negative-price "
  "intervals are retained; the average local price and net-demand trough is "
  "approximately 12:00 PDT; the average local price and net-demand peak is "
  "approximately 20:00 PDT; and price/net-demand correlation is approximately "
  "0.895.")
P("Negative-net-demand intervals average $2.86/MWh versus $29.88/MWh otherwise. "
  "The deepest 5% of net-demand intervals average $0.11/MWh and have negative "
  "prices 56% of the time. Real energy cost therefore already strongly favors "
  "execution in the duck-curve belly.")
H2("QP headroom gate")
TABLE(
    ["Scenario", "Status quo", "Spatial QP", "Joint QP", "Spatial", "Joint", "Temporal"],
    [
        ["US a–d", "$6.568M", "$6.092M", "$6.045M", "7.24%", "7.95%", "0.77%"],
        ["US e–h", "$6.219M", "$5.768M", "$5.637M", "7.26%", "9.36%", "2.27%"],
        ["Global a–d", "$6.598M", "$5.559M", "$5.525M", "15.75%", "16.26%", "0.60%"],
        ["Global e–h", "$6.276M", "$5.246M", "$5.185M", "16.42%", "17.38%", "1.16%"],
    ],
    "NO-TRAINING CLAIRVOYANT-QP HEADROOM (UNRESTRICTED-ROUTING UPPER BOUND)",
    widths=[1.05, 0.85, 0.85, 0.85, 0.75, 0.75, 0.75],
)
H2("Objective coefficient and demand-charge treatment")
P("The frozen primary coefficient is α=0.015. Calibrated only on a–d, "
  "the Status Quo grid-stress component is 18.2% of real energy cost in US and "
  "18.3% in Global: material but not dominant. Headroom conclusions remain "
  "stable for α∈{0,0.005,0.015,0.03}.")
P("The standardized $15/kW-cycle demand charge is excluded from the primary "
  "reward and reported as a secondary sensitivity. It is about $4.694M for "
  "Status Quo—large enough to dominate—and is not a real Dutch or Singapore "
  "tariff. Demand-aware training remains a separate extension.")
FIG(
    ENERGY_FIGS / "objective_sensitivity.png",
    "Fig. 3. Objective sensitivity on a–d calibration cells only: component share and "
    "clairvoyant headroom across α.",
    width=6.6,
)
H2("Structural attribution and rated-power sensitivity")
P("V2 uses one CAISO price level shifted in local time, so the v1 unequal-mean "
  "synthetic-price criticism no longer applies. The a–d-only ablations below "
  "diagnose interacting sources of spatial headroom; they are not an additive "
  "causal decomposition.")
TABLE(
    ["Condition", "US", "Global"],
    [
        ["Primary", "7.24%", "15.75%"],
        ["Energy only", "7.52%", "13.77%"],
        ["Pooled power", "4.14%", "14.64%"],
        ["Synchronous market", "5.67%", "5.67%"],
        ["Shifted market only", "4.36%", "12.58%"],
        ["Power heterogeneity only", "6.33%", "6.33%"],
        ["Fully equal control", "0.05%", "0.05%"],
    ],
    "SPATIAL-QP STRUCTURAL ABLATIONS (A–D CALIBRATION CELLS)",
    widths=[2.2, 1.0, 1.0],
)
P("Shifted market phase is the dominant Global lever; calibrated per-cell power "
  "heterogeneity materially increases US headroom. The latter makes the "
  "optimization non-degenerate but is a direct linear-model consequence, not "
  "a novel RL discovery. With fixed α, R=50–200 MW yields 7.18–7.34% US and "
  "14.81–17.35% Global headroom.")
FIG(
    ENERGY_FIGS / "structure_ablation.png",
    "Fig. 4. No-training structural and rated-power sensitivity on a–d calibration "
    "cells. Mechanisms interact; bars are not additive attributions.",
    width=6.6,
)
H2("Experimental deadline sensitivity")
P("ClusterData 2019 supplies no deadlines. The primary φ=1 gives H=2μ. "
  "Clairvoyant robustness at φ=0 (H=μ) yields 0.3–1.2% incremental temporal "
  "headroom; primary φ=1 yields 0.6–2.3%; loose φ=2 (H=3μ) yields 0.8–3.2%. "
  "Temporal value remains secondary across the sweep.")
H2("Figures")
FIG(
    ENERGY_FIGS / "reference_month.png",
    "Fig. 5. Energy model v2 reference month: real May-2025 CAISO day-ahead price, "
    "net demand, and solar.",
    width=6.6,
)
FIG(
    ENERGY_FIGS / "us_shifted_daily_profiles.png",
    "Fig. 6. Energy model v2 average daily profiles shifted across Pacific, Mountain, "
    "Central, and Eastern slots.",
    width=6.6,
)
FIG(
    ENERGY_FIGS / "global_shifted_daily_profiles.png",
    "Fig. 7. Energy model v2 average daily profiles shifted across Pacific, Central, "
    "Amsterdam, and Singapore slots.",
    width=6.6,
)
H2("Review interpretation")
P("The corrected signed-demand/equal-capacity model creates 8.0–17.4% "
  "optimistic unrestricted-routing joint headroom depending on scenario. "
  "Spatial diversity remains dominant. "
  "Incremental temporal headroom is positive but modest at 0.6–2.3%.")
H2("Explicit limitations and future work")
P("The primary experiment is a controlled CAISO archetype, not a real multi-"
  "market replay. Regional price levels are intentionally not re-averaged. "
  "Time-zone shifts use adjacent real boundary hours rather than a circular "
  "within-May wrap, so shifted monthly means can differ slightly. Five-minute "
  "RTM price is future robustness work. Historical 2019/2024 duck-curve "
  "comparison and extrapolation are future work, not additional training "
  "scenarios. Workload shapes are from May 2019 while the energy calendar is "
  "May 2025; this is an explicit counterfactual. OOF holds out workload cells, "
  "not the shared energy month. Routing is unrestricted across all slots; "
  "latency, residency, and movement are future work. MPC is future work; the "
  "QP is a clairvoyant diagnostic only.")
H2("Gate status")
P("The energy gate was consumed by the completed 80-model frozen OOF campaign. "
  "The joint-shaping headline criterion failed; only Global spatial PPO "
  "produced positive held-out optimizer CIs in both folds.",
  bold=True)

body_after_gate = doc.add_section(WD_SECTION.CONTINUOUS)
set_cols(body_after_gate, 1)

# ---------------- V. FROZEN V2 PROTOCOL ----------------
H1("V. END-TO-END FROZEN V2 EXPERIMENTAL PROTOCOL")
P("Figure 5 materializes the complete research chain for a reader unfamiliar "
  "with PPO training. Every arrow corresponds to a persisted, hashed source, "
  "transformation, scenario, model, or result artifact; no v2 result is selected "
  "on the held-out fold.")
FIG(
    OOF_FIGS / "end_to_end_pipeline.png",
    "Fig. 8. End-to-end v2 pipeline: public workload/power/market traces are "
    "transformed into an equal-proxy Gymnasium system, frozen PPO training, "
    "opposite-cell evaluation, and canonical thesis evidence.",
    width=6.6,
)
H2("A. Workload materialization")
P("ClusterData2019 instance_usage is joined to collection priority and "
  "aggregated into five-minute curves. Priority ≤115 supplies measured no-SLO "
  "batch; the remainder supplies service, with service+batch=aggregate exactly. "
  "Each own-cell-normalized shape maps to a unit-capacity 100 MW proxy. The "
  "synthetic generator is bypassed; fitted mean duration only parameterizes the "
  "experimental deadline.")
H2("B. Physical and market materialization")
P("PowerData2019 supplies per-cell idle/slope models. CAISO Today's Outlook "
  "supplies May-2025 five-minute signed net demand/solar, and OASIS NP15 supplies "
  "real hourly DAM price. The price/demand pair is shifted together across US "
  "and Global local clocks, retaining one controlled price level. Routing is "
  "unrestricted and therefore represents an optimistic upper bound.")
H2("C. State, action, and objective")
P("At each step PPO observes measured service, the current measured batch "
  "arrival, carried EDF pool/urgency, backlog, real price, signed net demand, "
  "solar, current load, and site context. It outputs service-routing fractions, "
  "batch-release rates, and batch-placement fractions. Service receives first "
  "capacity; only completed batch leaves its origin queue. The primary objective "
  "is real energy + α=0.015 positive-grid-stress cost + service/completion "
  "safeguards. Demand charge and 1h/3h ramp rate are independent outcomes.")
H2("D. Frozen training protocol")
TABLE(
    ["Element", "Frozen value"],
    [
        ["Algorithm", "PPO; MLP [128,128]; lr=3e-4"],
        ["Rollout / minibatch", "2,048 / 64"],
        ["Budget", "501,760 steps = 245 rollouts/model"],
        ["Discount", "γ=1"],
        ["Seeds", "101–110"],
        ["Fold A", "train a–d; test frozen policy on e–h"],
        ["Fold B", "train e–h; test frozen policy on a–d"],
        ["Configs", "US/Global × spatial-only/joint batch"],
        ["Total", "2 × 4 × 10 = 80 models"],
        ["Selection", "none; no validation/test-fold tuning"],
    ],
    "TABLE III. FROZEN ENERGY-MODEL V2 PPO PROTOCOL",
    widths=[1.8, 4.7],
)
P("The runner enforces a clean committed tree; hashes source, data, package "
  "versions, protocol, models, and logs; publishes each model atomically; and "
  "requires all 80 completion records before evaluation.")
H2("E. Held-out evaluation and statistics")
P("Every policy executes one deterministic complete held-out month. References "
  "are Status Quo, Round Robin, Drain Immediately (joint mode), and the "
  "clairvoyant QP diagnostic. The frozen 95% interval is a 20,000-resample "
  "percentile bootstrap over n=10 optimizer seeds. This interval describes "
  "optimizer variability, not workload- or market-population uncertainty; a "
  "Student-t sensitivity is also reported.")

# ---------------- VI. FROZEN V2 RESULTS ----------------
H1("VI. FROZEN ENERGY-MODEL V2 HELD-OUT RESULTS")
P("Frozen verdict — JOINT-SHAPING HEADLINE NOT SUPPORTED.", bold=True)
TABLE(
    ["Fold", "Config", "Savings [95% CI]", "+ seeds", "Feasible", "QP head", "PPO gap"],
    [
        ["a–d→e–h", "US spatial", "−0.29% [−0.76,+0.15]", "4/10", "10/10", "7.26%", "8.13%"],
        ["a–d→e–h", "US joint", "−4.30% [−12.45,+0.20]", "4/10", "3/10", "9.36%", "15.07%"],
        ["a–d→e–h", "Global spatial", "+0.90% [+0.34,+1.42]", "8/10", "10/10", "16.42%", "18.57%"],
        ["a–d→e–h", "Global joint", "+0.17% [−1.00,+1.19]", "6/10", "2/10", "17.38%", "20.84%"],
        ["e–h→a–d", "US spatial", "+0.07% [−0.27,+0.42]", "5/10", "10/10", "7.24%", "7.74%"],
        ["e–h→a–d", "US joint", "+0.04% [−0.59,+0.71]", "4/10", "2/10", "7.95%", "8.60%"],
        ["e–h→a–d", "Global spatial", "+1.19% [+0.81,+1.58]", "10/10", "10/10", "15.75%", "17.29%"],
        ["e–h→a–d", "Global joint", "+1.38% [+0.82,+1.87]", "9/10", "2/10", "16.26%", "17.77%"],
    ],
    "TABLE IV. HELD-OUT PPO SAVINGS VS STATUS QUO (10 SEEDS/FOLD)",
    widths=[0.85, 0.85, 1.85, 0.55, 0.65, 0.65, 0.65],
)
FIG(
    OOF_FIGS / "held_out_savings.png",
    "Fig. 9. Held-out seed savings, means, and frozen 95% optimizer-bootstrap "
    "intervals. Only Global spatial is positive in both folds.",
    width=6.6,
)
H2("A. The only robust learned success: Global spatial")
P("Global spatial PPO saves 0.90% and 1.19% on the two held-out folds, with "
  "complete service and positive optimizer-bootstrap intervals. Wider n=10 "
  "Student-t intervals remain positive: [0.24,1.55]% and [0.71,1.66]%. The "
  "effect is real within this frozen optimizer-seed experiment but small and "
  "optimistic because routing is unrestricted. PPO captures only 5.47% and "
  "7.55% of available QP savings.")
H2("B. US spatial savings are not established")
P("US a–d→e–h averages −0.29% [−0.76,+0.15] and its PPO mean also loses to "
  "Round Robin. The reverse fold averages +0.07% [−0.27,+0.42]. Thus US spatial "
  "is consistent with a small loss in one direction and indistinguishable from "
  "zero in the other.")
H2("C. Joint batch control fails the frozen criterion")
P("Joint PPO does not reliably improve over separately trained spatial PPO. It "
  "is worse in both US folds and Global a–d→e–h, and only 0.19% better in the "
  "remaining Global fold. This is not a clean estimate of pure temporal value "
  "because completion fails in 31/40 batch seeds. Only 9/40 seeds (2 configs × "
  "2 folds × 10) meet the 99.99% floor; no joint configuration reaches 4/10 "
  "feasible seeds in a fold.")
P("All 80 policies complete 100% of service. Global joint e–h→a–d seed 103 "
  "expires 0.889 normalized units. US joint a–d→e–h seed 101 leaves the largest "
  "terminal pool (7.690 units) and accumulates $2.289M of transient service-"
  "backlog cost, producing the −39.77% outlier. The QP proves temporal "
  "opportunity exists; PPO does not capture it reliably.")
P("This negative joint result is itself conditional on an omitted state "
  "variable and a frozen budget, not a clean statement that PPO cannot learn "
  "joint control: the frozen reward charged joint policies a terminal-"
  "completion liability without letting them observe episode position whenever "
  "the (default-off) demand charge was disabled, so the same partial state "
  "carried two different marginal completion costs late in the episode. "
  "Sec. VII adds the missing episode position and deadline buckets, re-screens optimizer and reward-weight "
  "hyperparameters, and trains at up to 2× this budget under a stricter "
  "safety-first gate — and still fails to produce a trustworthy joint "
  "controller, so the frozen conclusion above stands.")
FIG(
    OOF_FIGS / "qp_capture.png",
    "Fig. 10. Clairvoyant opportunity versus learned held-out savings. Large QP "
    "headroom does not translate into comparable PPO gains.",
    width=6.6,
)
H2("D. Secondary billing and physical-grid effects")
P("Spatial PPO lowers the standardized secondary demand charge by 1.4–3.0% "
  "across folds. Joint PPO raises it by 5.3–8.1%, showing that unstable temporal "
  "control can reduce the primary objective while worsening a site peak tariff. "
  "Ramp rate is not optimized: joint PPO worsens the rare maximum three-hour "
  "ramp across all sites in every fold by +1.09 to +3.05 MW, although typical "
  "p95 three-hour ramps improve. Spatial ramp effects are small and mixed.")
FIG(
    OOF_FIGS / "secondary_effects.png",
    "Fig. 11. Secondary demand-charge and physical ramp effects versus held-out "
    "Status Quo. Negative values indicate improvement.",
    width=6.6,
)
H2("E. Empirical conclusion and claim boundary")
P("The frozen campaign supports a narrow conclusion: one real CAISO archetype "
  "contains substantial optimistic spatial opportunity, and PPO captures a "
  "small reproducible part only in the Global spatial setting. It does not "
  "establish US savings or reliable joint spatio-temporal optimization. Claims "
  "are limited to optimizer-seed variability, one Google workload month, one "
  "CAISO energy month, equal 100 MW proxies, unrestricted routing, and the "
  "constructed Φ metric. This negative headline result is scientifically useful: "
  "it identifies batch-safe constrained control and stronger spatial policy "
  "optimization as the next algorithmic problems.")

# ---------------- VII. V3 RECOVERY STUDY ----------------
H1("VII. EXPLORATORY V3 STATE-REPAIR AND JOINT PPO RECOVERY STUDY")
P("Status — POST-HOC, EXPLORATORY, NON-HEADLINE. This section reports a "
  "protocol-frozen but non-selection-gated follow-up (env/protocols/"
  "v3_reward_sweep.yaml, parent protocol energy-model-v2-2025) that partially "
  "repairs a diagnosed state-observability defect in the frozen v2 joint controller and "
  "re-screens PPO under a stricter, pre-registered safety-first gate. It "
  "explicitly excludes model-predictive control and spatial-only retraining, "
  "and it does not overturn the frozen Sec. VI evidence, which remains the "
  "thesis's primary held-out result.", bold=True)
H2("A. Motivating defect and reframing of the v2 joint claim")
P("Diagnosis of the Sec. VI-C failures found that the frozen v2 joint reward "
  "charged terminal batch-completion liability (backlog and expiry weight) "
  "without exposing episode position to the policy whenever the primary "
  "demand-charge rate was zero — the default-off configuration used for every "
  "frozen headline number. Two states that differ only in how much episode "
  "remains therefore looked identical to the policy while carrying different "
  "true marginal costs of leaving work unfinished near the horizon. The v2 "
  "joint negative result is consequently conditional on this partially "
  "observable state representation and on one frozen 501,760-step training "
  "budget — not clean evidence that PPO cannot learn joint control in this "
  "environment. This section tests that conditional claim directly by fixing "
  "the state and enlarging the budget, while leaving every other frozen v2 "
  "modeling choice (equal 100 MW proxies, unrestricted routing, Φ objective, "
  "the same US/Global scenarios) untouched.")
H2("B. V3 state additions, action, and objective changes")
P("Two observation additions address the hidden episode horizon and coarse "
  "deadline context, gated behind opt-in flags so "
  "frozen v2 models are unaffected: pre-action episode progress and remaining-"
  "horizon fractions t/(T−1) and (T−1−t)/(T−1) (observe_episode_progress), and "
  "a per-site histogram of batch-pool mass across deadline buckets edged at "
  "{1,3,6,12,24} steps plus the current arrival's own bucket "
  "(deadline_bucket_edges) — six bins per site instead of one pooled backlog "
  "scalar. Both are computed pre-action; Sec. VII-G documents a remaining "
  "deadline-boundary observation defect found after the run. The action space, "
  "decode (softmax/sigmoid, "
  "±3 logit bound), and per-step objective (Eq. 10) are unchanged; only joint "
  "temporal+spatial control is retrained (spatial-only retraining and MPC are "
  "protocol-excluded, Sec. VII intro). Evaluation always scores the full-"
  "dollar objective at fixed guard weights — service-backlog and batch-"
  "completion weight 1000/1000, matched to the ≈$626/normalized-unit economic "
  "floor computed from the largest one-step saving obtainable by dropping "
  "served work — so the metric used to rank policies never changes across the "
  "sweep. Only the TRAINING-only reward weights (service-backlog, batch-"
  "completion, and an optional urgency potential) vary between candidates; "
  "this separates a genuine hyperparameter/incentive sweep from a moving "
  "scoreboard.")
H2("C. Staged protocol: rollout-aligned budgets, seeds, and safety-first selection")
TABLE(
    ["Stage", "Budget (rollouts)", "Seeds", "Sweep axis", "Promotes"],
    [
        ["Round 1", "151,552 (74)", "201–203 (3)", "GAE λ∈{.95,.99,1.0}; KL "
         "target∈{none,.02}; LR sched∈{fixed,linear}; reward mode∈{full,"
         "idle-subtracted}; obs. normalization∈{off,on} — 6 candidates R0–R5",
         "1 candidate/region"],
        ["Round 2", "301,056 (147)", "201–205 (5)", "Training-only reward "
         "weights on the Round-1 winner: service-backlog/batch-completion/"
         "urgency-potential — 4 variants P0–P3", "1 variant/region"],
        ["Full development", "501,760 (245)", "201–210 (10)", "Winning combo "
         "only — reproduces the frozen v2 budget under the augmented state",
         "gate check"],
        ["Budget scaling", "151,552 / 501,760 / 1,003,520 (74/245/490)",
         "compare 201–205 (5); replicate 201–210 (10)", "Winning combo only "
         "— diagnoses whether 501,760 steps was compute-limited",
         "1 budget/region"],
    ],
    "TABLE V. STAGED V3 PROTOCOL (a–d DEVELOPMENT CELLS)",
    widths=[1.1, 1.55, 1.05, 2.2, 0.9],
)
P("Safety-first selection is lexicographic and identical in structure at "
  "every stage, but step (1) is evaluated against the seed count run AT THAT "
  "STAGE, not a fixed count of 10: (1) all seeds in the current stage safe — "
  "3 seeds in Round 1, 5 seeds in Round 2 and in the budget-scaling comparison "
  "sweep, 10 seeds in full development and in the final budget-scaling "
  "replication — (2) most safe seeds, (3) lowest worst-seed total cost, "
  "(4) lowest mean total cost, (5, budget stage only) lower training budget as "
  "a final tie-break. A seed is SAFE only if it clears every one of: "
  "service-completion floor ≥0.999999999, batch-completion floor ≥0.9999, "
  "expired-work tolerance ≤1e-9, terminal-pool fraction tolerance ≤1e-4, and "
  "maximum transient service backlog ≤0.25 normalized units. The pre-"
  "registered final gate additionally requires ALL 10 seeds safe and a "
  "positive 95% optimizer-bootstrap CI vs. Status Quo; failing either reports "
  "the recovery as unsuccessful without activating any controller.")
TABLE(
    ["Region", "Cand.", "GAE λ", "KL", "LR", "Reward mode", "Norm.", "Mean sav.", "Worst sav."],
    [
        ["US", "R0", "0.95", "—", "fixed", "full", "off", "−0.101%", "−0.787%"],
        ["US", "R1", "0.99", "—", "fixed", "full", "off", "−0.239%", "−0.770%"],
        ["US", "R2", "1.00", "—", "fixed", "full", "off", "−0.131%", "−0.889%"],
        ["US", "R3 ★", "0.99", "0.02", "linear", "full", "off", "+0.171%", "−0.072%"],
        ["US", "R4", "0.99", "0.02", "linear", "idle-sub.", "off", "−0.367%", "−1.697%"],
        ["US", "R5", "0.99", "0.02", "linear", "idle-sub.", "on", "+0.047%", "−0.638%"],
        ["Global", "R0 ★", "0.95", "—", "fixed", "full", "off", "+1.893%", "+0.755%"],
        ["Global", "R1", "0.99", "—", "fixed", "full", "off", "+0.950%", "−0.130%"],
        ["Global", "R2", "1.00", "—", "fixed", "full", "off", "−1.185%", "−2.869%"],
        ["Global", "R3", "0.99", "0.02", "linear", "full", "off", "+0.998%", "+0.322%"],
        ["Global", "R4", "0.99", "0.02", "linear", "idle-sub.", "off", "+0.903%", "+0.537%"],
        ["Global", "R5", "0.99", "0.02", "linear", "idle-sub.", "on", "−0.035%", "−0.599%"],
    ],
    "TABLE VI. ROUND-1 OPTIMIZER/STATE SCREEN, 3 SEEDS/CANDIDATE (★=PROMOTED; "
    "ALL CANDIDATES 0/3 SAFE)",
    widths=[0.65, 0.5, 0.45, 0.4, 0.45, 0.75, 0.4, 0.65, 0.65],
)
P("No Round-1 candidate is safe on any seed — the state augmentation alone does not "
  "restore completion feasibility. R3 (GAE λ=0.99, KL target 0.02, linear LR "
  "decay, full reward mode) is promoted for US on mean/worst savings; R0 "
  "(GAE λ=0.95, fixed LR, no KL target) is promoted for Global on the same "
  "criteria. Both promotions are ties broken by economics only, since no "
  "candidate clears the safety floor.")
TABLE(
    ["Region", "Variant", "Backlog wt.", "Completion wt.", "Potential wt.", "Mean sav.", "Worst sav."],
    [
        ["US", "P0", "1000", "1000", "0", "+0.191%", "−1.952%"],
        ["US", "P1 ★", "700", "1000", "0", "+0.547%", "+0.303%"],
        ["US", "P2", "700", "2000", "0", "+0.401%", "−0.473%"],
        ["US", "P3", "700", "1500", "250", "+0.452%", "−0.614%"],
        ["Global", "P0", "1000", "1000", "0", "+1.654%", "−1.070%"],
        ["Global", "P1", "700", "1000", "0", "+1.586%", "−0.525%"],
        ["Global", "P2", "700", "2000", "0", "+2.014%", "+0.182%"],
        ["Global", "P3 ★", "700", "1500", "250", "+1.167%", "+0.306%"],
    ],
    "TABLE VII. ROUND-2 REWARD-WEIGHT SCREEN ON THE ROUND-1 WINNER, 5 "
    "SEEDS/VARIANT (★=PROMOTED; ALL VARIANTS 0/5 SAFE)",
    widths=[0.65, 0.65, 0.85, 0.95, 0.85, 0.65, 0.65],
)
P("Combo R3_P1 (US) and R0_P3 (Global) carry forward to full development and "
  "budget scaling. P3's urgency-potential term has the best worst-seed savings "
  "for Global among safety-tied variants (all 0/5 safe) but is not "
  "categorically safer; P1 wins US on worst-seed cost despite reducing the "
  "backlog weight below the P0 baseline, illustrating that within this sweep "
  "reward-weight choice trades off mean economics against tail cost rather "
  "than buying completion safety outright.")
H2("D. Three-point matched-config budget curve")
TABLE(
    ["Region", "Combo", "Budget", "Mean sav.", "Worst sav.", "Safe seeds", "Notes"],
    [
        ["US", "R3_P1", "151,552", "+0.205%", "−0.513%", "1/5", "—"],
        ["US", "R3_P1", "501,760", "+0.350%", "−0.676%", "0/5", "backlog>0"],
        ["US", "R3_P1", "1,003,520", "−0.241%", "−1.014%", "0/5", "16.7 units "
         "expired; 2.16% action saturation"],
        ["Global", "R0_P3", "151,552", "+0.889%", "−1.020%", "0/5", "—"],
        ["Global", "R0_P3", "501,760", "+1.231%", "+0.138%", "0/5", "backlog "
         "1.23 units"],
        ["Global", "R0_P3", "1,003,520", "+1.764%", "+0.996%", "0/5", "—"],
    ],
    "TABLE VIII. MATCHED-CONFIG BUDGET CURVE, 5 SEEDS (201–205) PER POINT",
    widths=[0.6, 0.65, 0.85, 0.65, 0.65, 0.65, 1.6],
)
P("More compute helps Global economics monotonically (+0.889% → +1.231% → "
  "+1.764% as budget rises from 151,552 to 1,003,520 steps) but not safety: "
  "0/5 seeds are safe at every Global budget on this matched-seed comparison. "
  "US shows the opposite compute pathology: mean savings rise from +0.205% to "
  "+0.350% and then reverse to −0.241% at 1,003,520 steps, where the policy "
  "also begins expiring batch work (16.7 normalized units) and saturating "
  "actions (2.16% of steps) — degradation, not diminishing returns. Extra "
  "optimizer budget therefore helps Global's economics without helping its "
  "safety, and actively hurts US at the largest budget tested; there is no "
  "budget in this sweep at which unconstrained joint PPO becomes a safe "
  "controller.")
H2("E. Final safety-first-selected configuration and gate result (a–d)")
TABLE(
    ["Region", "Combo", "Budget", "Safe seeds", "Mean sav.", "Worst sav.", "Optimizer 95% CI (USD)", "All-seed +"],
    [
        ["US", "R3_P1", "151,552", "1/10", "+0.051%", "−1.057%",
         "[−22,193.78, +28,226.55]", "No"],
        ["Global", "R0_P3", "1,003,520", "1/10", "+2.032%", "+0.996%",
         "[+103,383.64, +167,438.74]", "Yes"],
    ],
    "TABLE IX. FINAL SAFETY-FIRST GATE, 10 SEEDS (201–210), a–d DEVELOPMENT CELLS",
    widths=[0.55, 0.55, 0.7, 0.65, 0.6, 0.6, 1.55, 0.6],
)
P("The lexicographic rule selects 151,552 steps for US (uniquely 1/5 safe "
  "among the three matched budgets) and 1,003,520 steps for Global (all three "
  "budgets tied at 0/5 safe on the comparison seeds, broken by lowest worst-"
  "seed and mean cost). Re-validated at 10 seeds, US R3_P1 is 1/10 safe with "
  "mean savings +0.051% and an optimizer-bootstrap CI that crosses zero "
  "([−$22,193.78, +$28,226.55]) — economically indistinguishable from Status "
  "Quo. Global R0_P3 is also only 1/10 safe despite +2.032% mean savings and a "
  "CI strictly above zero ([+$103,383.64, +$167,438.74], all 10 seeds "
  "individually positive): the pre-registered gate requires ALL seeds safe, "
  "so Global fails on completion even though it would pass on economics alone. "
  "Both regions therefore fail the final gate (protocol failure_outcome: "
  "report the recovery as unsuccessful; do not activate a spatial-only, "
  "temporal-only, or MPC controller).")
H2("F. Descriptive e–h transfer — not confirmatory, not a headline")
P("Cells e–h were already exposed during the frozen v2 campaign (Sec. V-E), so "
  "evaluating the two safety-first-selected v3 combos there — US R3_P1 "
  "trained at its selected budget of 151,552 steps and Global R0_P3 trained "
  "at its selected budget of 1,003,520 steps (Sec. VII-E) — is a one-time "
  "descriptive check, not fresh confirmatory data and not headline-eligible "
  "(final_eh_transfer.json: headline_eligible=false). It is reported for "
  "completeness only.")
TABLE(
    ["Region", "Combo", "Training budget", "Safe seeds", "Mean savings", "Worst savings"],
    [
        ["US", "R3_P1", "151,552", "7/10", "−0.155%", "−0.760%"],
        ["Global", "R0_P3", "1,003,520", "5/10", "+2.826%", "+1.106%"],
    ],
    "TABLE X. DESCRIPTIVE e–h TRANSFER OF THE a–d-SELECTED-BUDGET MODELS, 10 "
    "SEEDS (NON-CONFIRMATORY)",
    widths=[0.65, 0.65, 0.95, 0.75, 0.95, 0.95],
)
P("Transfer is not consistently better than the a–d gate: US mean savings turn "
  "negative on e–h even though more seeds individually clear the safety floor "
  "(7/10 vs. 1/10), and Global remains only half-safe. This inconsistency is "
  "additional descriptive evidence against treating either combo as a settled "
  "joint controller, but because e–h is not fresh data it cannot itself "
  "confirm or refute the a–d gate result.")
H2("G. Documented limitations and claim boundary")
P("V2 reward-scale confound (not resolved by v3). The frozen v2 protocol "
  "trained spatial-only policies at reward_scale=1.0 and completion-guarded "
  "joint (batch) policies at reward_scale=1e-4, with an identical learning "
  "rate (3e-4) across both configurations (env/protocols/v2_2025.yaml). The "
  "two controller classes were therefore optimized against training signals "
  "of very different numerical magnitude even though both are scored on the "
  "same dollar objective at evaluation time, so v2's spatial-vs-joint "
  "comparison mixes a genuine policy-class difference with an unmatched "
  "training scale. V3 does not resolve this confound: it trains joint "
  "(temporal+spatial) policies only, at one fixed reward_scale throughout "
  "Rounds 1–2 and budget scaling (Sec. VII-B), and never introduces or "
  "repeats a spatial-only run at a matched scale for comparison. This section "
  "therefore cannot and does not settle the historical v2 spatial-vs-joint "
  "cross-control comparison; it only asks whether joint PPO, evaluated on its "
  "own terms, can pass a stricter safety-first bar.")
P("Equal-capacity limitation (unchanged from v2). Every site remains an equal "
  "100 MW proxy. This isolates workload, market-phase, and calibrated power-"
  "model effects, but it also removes real fleet-size heterogeneity and may "
  "understate or overstate US spatial opportunity; it is an explicit "
  "controlled-system limitation carried into v3 unchanged, not a v3-specific "
  "artifact.")
P("Negative-net-demand objective caveat (unchanged from v2). Φ is zero "
  "whenever net demand is negative (Eq. 4), so nothing in the reward "
  "discourages concentrating joint routing and drain during those intervals; "
  "the v3 policies' behavior there is only observed and audited (Sec. VI-D "
  "ramp KPIs), never separately regularized, and this caveat applies to every "
  "v3 result in this section exactly as it applies to v2.")
P("Deterministic negative-net-demand probe (a–d, ten deterministic seed "
  "policies per region). To audit rather than merely flag the preceding "
  "caveat, we reconstructed the a–d evaluation environment and reran all ten "
  "selected-policy seeds per region with domain randomization disabled — US "
  "R3_P1 at 151,552 steps and Global R0_P3 at 1,003,520 steps — contrasting "
  "steps with at least one site at negative net demand against all other "
  "steps (Table XI).")
TABLE(
    ["Region", "Neg.-step share", "Drain, ≥1 neg.", "Drain, else",
     "Clear., ≥1 neg.", "Clear., else", "Route share, neg. (policy)",
     "Route share, neg. (uniform)", "Neg. steps, drain>90%"],
    [
        ["US", "23.89%", "50.91%", "51.32%", "50.82%", "50.79%", "61.11%",
         "61.18%", "0.0%"],
        ["Global", "48.25%", "52.00%", "52.12%", "50.15%", "50.21%", "33.26%",
         "30.05%", "0.0%"],
    ],
    "TABLE XI. DETERMINISTIC NEGATIVE-NET-DEMAND PROBE, ALL-10-SEED SELECTED "
    "a–d POLICIES",
    widths=[0.4, 0.6, 0.6, 0.55, 0.65, 0.6, 0.85, 0.85, 0.7],
)
P("The uniform benchmark is the mean fraction of destinations that are "
  "themselves at negative net demand on the same any-negative steps, i.e. "
  "the routed share a destination-blind policy would achieve by "
  "construction. US policy route share to negative-demand destinations "
    "(61.11%) is essentially indistinguishable from — if anything, marginally "
  "below — that uniform benchmark (61.18%): US routing shows no systematic "
  "preference for or against negative-demand destinations. Global policy "
  "route share (33.26%) exceeds its uniform benchmark (30.05%) by 3.21 "
  "percentage points, a modest but real tilt toward negative-demand "
  "destinations. Neither region shows increased drainage on those steps: "
  "mean drain is, if anything, slightly LOWER on negative-demand steps than "
  "otherwise in both regions (US 50.91% vs. 51.32%; Global 52.00% vs. "
  "52.12%), realized pool clearance is effectively unchanged between the two "
  "conditions in both regions (US 50.82% vs. 50.79%; Global 50.15% vs. "
  "50.21%), and the share of negative-demand steps with mean drain above 90% "
  "is 0.0% for both US and Global — neither policy blasts through capacity. "
  "This is a deterministic, descriptive audit of ten deterministic seed "
  "policies per region on the fixed evaluation trace, not a causal or "
  "confirmatory claim about why either policy routes as it does, and it does "
  "not alter the Sec. VII-E gate result.")
P("Demand-charge shorter-period guard and telescoping regression tests "
  "(unchanged from v2, exercised by v3's own smoke tests). A demand-charge "
  "period shorter than the episode raises a hard error unless "
  "allow_multiple_demand_charge_periods=True is passed explicitly, preventing "
  "an accidental per-sub-period tariff from silently multiplying the reference "
  "charge. scripts/smoke_test_demand_charge.py numerically asserts two "
  "already-published identities to machine precision: (i) incremental "
  "per-step demand charges sum to c×max_t g_t over each complete billing "
  "period, and (ii) summed arrivals minus completions equals expired work "
  "plus the terminal batch pool — an accounting identity, not a reward "
  "potential. scripts/smoke_test_ppo_v3.py separately and independently "
  "regression-tests that the new v3 urgency-potential shaping term telescopes "
  "under γ=1 to a policy-independent constant fixed only by the initial "
  "batch-pool state, so potential-based shaping cannot alter the undiscounted "
  "return regardless of which actions are taken; both suites pass for every "
  "configuration used in this section.")
P("Remaining deadline-boundary observation defect. At step t, the pre-action "
  "observation can include carried pool entries with deadline_step ≤ t even "
  "though the transition expires those entries before service. Selected US "
  "training had expiry in 0/10 seeds (0.000 units total), while selected Global "
  "training had expiry in 9/10 seeds (127.597 units across all randomized "
  "episodes). Every selected a–d and descriptive e–h evaluation seed had zero "
  "expiry, so the reported evaluation failures are terminal-pool failures rather "
  "than direct boundary-expiry artifacts; nevertheless, v3 is not a fully repaired "
  "state formulation. The next protocol must compute actionable pool size, "
  "urgency, and deadline buckets only from entries with deadline_step > t, "
  "optionally expose unavoidable due-now mass separately, and retrain before any "
  "positive state-repair claim. Frozen v3 semantics and results remain unchanged.")
P("Hard vs. soft constraints and the sigmoid reachability ceiling. Every "
  "safety property enforced here — service completion, batch completion, "
  "zero expiry, bounded backlog — remains a SOFT, reward-shaped penalty; none "
  "is a hard constraint on the action, which is why 31/40 v2 seeds and, now, "
  "9/10 or more v3 seeds per region can violate it despite heavy shaping. Only "
  "the batch DRAIN head is sigmoid-decoded; service routing and batch spatial "
  "placement are softmax-decoded shares that already sum to exactly 1 and "
  "carry no endpoint-reachability problem. For the drain head, the ±3 "
  "action-logit bound caps the reachable rate at sigmoid(3)=95.257% (and "
  "floors it at sigmoid(−3)=4.743%, Sec. III-C): a policy can request "
  "draining a site's entire batch pool in one step only up to 95.257% of it, "
  "never exactly 100%, regardless of training. Raising the bound narrows but "
  "never closes this gap — sigmoid(10)=99.995% already exceeds the 99.99% "
  "batch-completion floor used for the safety gate — yet the sigmoid's open "
  "range (0,1) means no finite logit bound reaches EXACT full drain. Because "
  "exact 0–100% reachability, not merely a value above 99.99%, is what a "
  "completion guarantee requires, merely widening the sigmoid bound is "
  "insufficient on its own: the next step (Sec. VIII) is not a larger action "
  "bound or an MPC comparator (both remain excluded by this protocol) but a "
  "non-MPC joint feasibility/action-projection layer — a hard decoder or a "
  "post-hoc override on the drain head with true 0/1 endpoints — that maps "
  "any learned action onto the nearest feasible one with exact reachable "
  "endpoints.")
P("Everything else about the v2 protocol continues to apply unchanged: ramp "
  "rate and the standardized demand charge remain independent, audited KPIs "
  "rather than reward terms (Sec. III-A); unrestricted cross-site routing "
  "remains an explicit optimistic assumption, not a deployment claim (Sec. "
  "III-C); and this section makes no MPC claim of any kind — MPC is "
  "structurally excluded from the v3 protocol, exactly as in v2.")
H2("H. Conclusion of the recovery study")
P("Repairing the diagnosed state-observability defect, re-screening six "
  "optimizer/state variants and four reward-weight variants per region, and "
  "training at up to 2× the frozen v2 budget does not produce a safety-first-"
  "gate-passing joint controller: the best US configuration remains "
  "economically indistinguishable from Status Quo (CI crossing zero) and the "
  "best Global configuration, despite a real positive economic effect, fails "
  "on completion safety with only 1/10 seeds safe. More compute measurably "
  "helps Global's economics and measurably hurts US's at the largest budget "
  "tested, but no amount tried buys joint safety. The evidence therefore "
  "reframes, but does not reverse, Sec. VI's negative joint finding: "
  "unconstrained PPO remains unsuitable as a trustworthy joint spatio-temporal "
  "controller under this model, whether or not it can observe episode "
  "position, and the appropriate next step is safety-constrained action "
  "projection (Sec. VIII), not a larger sweep of the same unconstrained "
  "policy class.")

# ---------------- V. EXPERIMENTAL SETUP ----------------
EMIT_CONTENT = False
H1("APPENDIX A. ARCHIVED ENERGY-MODEL V1 EXPERIMENTAL SETUP")
P("This section and Appendix B preserve the retired mixed/synthetic energy-model "
  "v1 campaign for auditability. They are not final v2 evidence and must be "
  "replaced after the frozen v2 campaign.", bold=True)
H2("A. Agents and baselines")
P("PPO (Stable-Baselines3, MLP 128×128, lr 3e-4, 500k steps) acts in the "
  "continuous 3N-dimensional space. Its observation includes per-DC dynamic "
  "state (demand, price, net demand, pool/backlog) AND static site context "
  "(per-DC idle power, slope, capacity, deferrable fraction) — the latter "
  "added so that per-site heterogeneity is exploitable as a function of "
  "observed parameters rather than memorized by slot (Appendix B-H). Two DQN "
  "variants share hyperparameters (MLP 256×256, replay 100k, target update "
  "1k): a 759-action routing grid, and a 48-action compact flattened index "
  "inspired by CFWS's encoding philosophy, decoding to (source DC, destination "
  "DC, drain level) [6]. It is not a CFWS reproduction. Nine heuristics "
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
  "modeling errors of Sec. VIII.")

# ---------------- VI. RESULTS ----------------
H1("APPENDIX B. ARCHIVED ENERGY-MODEL V1 RESULTS")
TABLE(["Config", "SQ", "Trough", "DQN", "PPO (±sd)", "ΔSQ", "QP", "gap"],
      [["US spatial", "9.85", "11.23", "9.81", "9.57±.03", "2.8%", "9.46", "1.6%"],
       ["US batch", "9.85", "10.09", "9.89", "9.59±.08", "2.7%", "9.44", "1.3%"],
       ["Glob spatial", "14.06", "14.32", "13.35", "12.62±.25", "10.3%", "11.95", "5.7%"],
       ["Glob batch", "14.06", "13.76", "13.46", "12.25±.14", "12.9%", "11.89", "3.1%"]],
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
  "within 1.3–5.7% of a clairvoyant convex-QP lower bound (Appendix B-F) — so "
  "little headroom remains to ANY policy, causal or not. Savings scale with "
  "exploitable structure: modest under US-only diversity, large under global "
  "price/timezone spread. PPO also serves 100% of service demand with zero "
  "deadline violations (Appendix B-E) and the flattest, lowest fleet draw "
  "(Fig. 4).")
FIG(FIGS / "fig4_profile.png",
    "Fig. 4. Fleet power profile, US batch (first 4 days): PPO serves the same "
    "work at lower, flatter draw.")
H2("B. Historical v1 power-heterogeneity interpretation")
P("Per-cell power calibration makes marginal work cheaper where fitted slope is "
  "lower because idle power is sunk. Historical PPO routed accordingly, but v1 "
  "did not isolate that mechanism from unequal synthetic regional prices. It is "
  "a direct consequence of the calibrated linear model—not a novel RL "
  "discovery—and its causal contribution is not assigned from v1 results.")
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
  "+6% estimates were artifacts L1–L3, Sec. VIII.)")
H2("E. Deadline and backlog audit")
P("Because every policy is scored on the shaped objective (Eq. 10), we audit "
  "that PPO's savings are genuine scheduling, not penalty-dodging. Across all "
  "ten PPO batch-mode seeds: service served / demand = 1.0000, terminal "
  "backlog = 0, terminal batch pool ≤ 0.95 units (≤0.025% of arrivals), ZERO "
  "expired batch, peak transient backlog ≤ 3.9 units, and backlog penalty ≤1%. "
  "Default-off evaluation retains that finite-horizon convention for model "
  "compatibility and reports the tail explicitly. Demand-enabled runs observe "
  "full-episode progress and charge terminal pool like expiry, closing the "
  "economically material tariff loophole. The historical tail is <$238 "
  "(<0.003% of cost), so rankings do not change, but 'all batch completed' was "
  "too strong.")
H2("F. Optimality gap")
P("The environment is convex in the serving decisions (linear power and energy "
  "cost, convex-quadratic peak penalty, linear queue/backlog dynamics), so a "
  "clairvoyant planner with full-episode foresight and fluid allocation solves "
  "a quadratic program whose optimum LOWER-BOUNDS any policy. We assemble it "
  "directly for the Clarabel solver with the simulator's service backlog, "
  "cumulative soft-expiry accounting, and optional demand-charge epigraph. The "
  "manual sparse assembly matches CVXPY below 10⁻⁸ relative and solves the "
  "four default-off configs in 2–15 s. PPO's gap is 1.6% (US spatial), 1.3% "
  "(US batch), 5.7% (Global spatial), and 3.1% (Global batch). The bound is generous — "
  "it has perfect foresight and no causality or action-parameterization "
  "constraints — so the true gap to the best ACHIEVABLE causal policy is "
  "smaller still. A useful by-product: the QP's spatial-only and batch optima "
  "differ by only ~0.2–0.5%, confirming the intrinsic value of temporal "
  "flexibility at these deferrable fractions is small (Appendix B-D).")
H2("G. The demand charge: post-hoc audit and trained extension")
P("Φ is a grid-stress shadow price, not the operator's tariff (Sec. III-A). "
  "Because a demand charge is the single largest line item that our objective "
  "omits, we quantify it for every policy from the grid-draw traces, at a "
  "reference c = $15/kW for the complete 8,917-step study billing cycle, "
  "WITHOUT putting it in the reward — so the policies compared are exactly "
  "those of Appendix B-A. Billing is per meter, so the billed quantity is "
  "Σᵢ maxₜ gᵢ,ₜ, not the fleet coincident peak.")
P("The charge is large: $4.69M per study cycle under the status quo, equal to "
  "39% of that scenario's $12.06M energy charge, 28% of the combined energy-"
  "plus-demand bill, and 2.3× the Φ term already in the objective. PPO reduces "
  "it to $4.32M (−8.0%, Global spatial+temporal) and $4.35M (−7.3%, US "
  "spatial+temporal) as a SIDE EFFECT of routing for energy price, with the "
  "term absent from its reward. These are reference-rate sensitivity results, "
  "not bill reconstructions.")
P("Two findings temper the obvious next step of adding it to the reward. "
  "First, the headroom is small, because most of the charge is not shapeable: "
  "sites never power off, so Σᵢ Pᵢⁱᵈˡᵉ·R = 189.8 MW = $2.85M is irreducible — "
  "66% of the best charge any policy could achieve. Solving for the "
  "peak-minimizing feasible allocation (the clairvoyant LP collapses to a "
  "greedy lowest-sᵢ·R fill, since only the busiest step binds) gives a floor of "
  "267.1 MW = $4.01M, so the entire prize is $0.69M and PPO already captures "
  "about half of it incidentally. What remains — roughly $0.31M, 2.5% of J — "
  "sits inside our seed-to-seed spread and would not survive as a headline "
  "result. Second, a demand charge would PENALIZE part of PPO's current "
  "strategy: concentrating load on the cheap, efficient sites raises their "
  "site peaks (Global batch: US-West +9%, US-Central +15%) even as EU and Asia "
  "fall, so a retrained policy would trade energy saving for peak relief "
  "rather than adding to it.")
P("The genuinely interesting consequence is for the temporal lever. Energy "
  "arbitrage pays only if timing is right CONTINUOUSLY for a month; clipping a "
  "demand charge pays if timing is right at the binding interval. Under the "
  "default-off objective the corrected QP values temporal flexibility at only "
  "0.19% (US) to 0.52% (Global). Adding the guarded $15/kW US-reference tariff "
  "raises the spatial-to-batch QP improvement to about 1.0–1.1% and lowers its "
  "optimal billed peak by about 9.6 MW in both scenarios, with numerical-zero "
  "expiry. The corrected environment uses the full episode as one billing "
  "cycle, resets period state before the next action, exposes the running peak "
  "and cycle context, and requires γ = 1 when the term is enabled so the "
  "incremental reward remains exactly the final billed maximum. Tariff-aware "
  "backlog/expiry floors prevent dropping work, reward scaling conditions RL, "
  "and dense arrival-minus-completion shaping preserves unfinished-work cost.")
TABLE(
    ["Config", "QP", "SQ", "Best feasible", "PPO", "PPO peak"],
    [
        ["US spatial", "13.820", "14.542", "RR 14.388", "14.396", "304.1"],
        ["US spatial+temp", "13.676", "14.545", "Drain 14.391", "14.561", "312.1"],
        ["Global spatial", "16.290", "18.756", "PPO 18.283", "18.283", "304.3"],
        ["Global spatial+temp", "16.139", "18.759", "Drain 18.641", "18.798", "311.3"],
    ],
    "DEMAND-AWARE SINGLE-SEED RESULTS (M$, PEAK IN MW)",
    widths=[0.82, 0.48, 0.48, 0.74, 0.5, 0.52],
)
P("These seed-42 runs are a negative but informative result. Demand-aware PPO "
  "reduces cost 1.0% vs. status quo in US spatial and 2.5% in Global spatial, "
  "but Round Robin narrowly wins US spatial. Adding the temporal head does not "
  "capture the QP's ~1% temporal opportunity: Drain Immediately is best in both "
  "batch scenarios, while PPO is 0.1–0.2% above status quo and 6.5–16.5% above "
  "the clairvoyant bound. Both DQN encodings are less reliable; incomplete-work "
  "penalties dominate several batch runs. Thus the tariff matters and the QP "
  "shows shapeable value, but 500k-step single-seed PPO/DQN does not establish a "
  "successful temporal demand-charge optimizer. These results are reported "
  "separately from the five-seed default-off headline.")
H2("H. Generalization: transfer tracks observability")
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
  "transfers at +5.2%. On an OBSERVED shift — US net demand swapped to real "
  "EIA-930 May-2024 while non-US sites use matching-year documented "
  "approximations, with prices fixed — the SAME frozen policies transfer "
  "near-perfectly (+3.1–13.2%, within "
  "~0.5 point of their on-training result). Generalization succeeds along the "
  "axis the policy can observe and fails along the one it cannot; this both "
  "answers the calendar-memorization concern for the demand signal and "
  "prescribes the fix (observe + randomize static context).")
FIG(FIGS / "fig6_generalization.png",
    "Fig. 6. Held-out transfer. Red: unobserved cell shift breaks context-only "
    "policies. Green: domain randomization restores it. Blue: the observed "
    "net-demand-year shift transfers without any special treatment.")
H2("I. Robustness to movement cost")
P("The archived v1 sweep charged service movement post hoc but omitted batch "
  "movement. That omission understates total movement cost and makes the result "
  "optimistic; it is not a conservative bound or valid robustness evidence. "
  "V2 does not promote its break-even values and instead frames unrestricted "
  "routing as an optimistic upper-bound assumption.")
FIG(FIGS / "fig7_movement_cost.png",
    "Fig. 7. PPO savings vs. per-unit inter-site movement cost (symlog x). "
    "Global routing keeps paying well past realistic egress prices.")

# ---------------- APPENDIX C. ARCHIVED LESSONS ----------------
H1("APPENDIX C. ARCHIVED LESSONS LEARNED AND THREATS TO VALIDITY")
P("Three modeling errors each inverted the experimental ranking while present; "
  "we report them as first-class results because any deferral-with-deadlines "
  "simulator can reproduce them.")
P("L1 — Deadline penalties must scale with the value of deferred work. A "
  "fixed expiry weight λ_x = 2 made expiring a unit ~75× cheaper than serving "
  "it (~$150 of energy); the cost-optimal policy was to discard batch, and "
  "aggressive-expiry heuristics beat PPO. Because the penalty is linear in "
  "expiry, any rollout can be re-ranked analytically as cost(λ_x) = "
  "(cost − λ₀·x) + λ_x·x (exact for a fixed rollout; retrained agents change "
  "behavior with λ_x); the ranking inverts at λ_x ≈ 183. We set λ_x = 250, "
  "safely above both the ~$150 serving cost and that inversion point.")
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
P("Threats to validity. (i) The active v2 experiment is a controlled CAISO "
  "archetype, not a real multi-market replay; regional price levels and demand "
  "shapes are intentionally held to one source. (ii) Hourly DAM price is held "
  "stepwise over twelve controller intervals; five-minute RTM is future "
  "robustness work. (iii) The continuous local-wall-time shift uses adjacent "
  "real boundary hours, so monthly slot means can differ slightly; Singapore's "
  "0.59% premium is a boundary effect, not a regional markup. (iv) Workload "
  "shapes are May 2019 while energy is May 2025, an explicit cross-year "
  "counterfactual. (v) Cell locations are anonymized; the geographic assignment "
  "is a modeling choice justified by documented inter-cell heterogeneity [1]. "
  "(vi) The 100 MW magnitude bridges cell scale to grid relevance and retains "
  "a price-taker assumption. (vii) The measured tier curves make episodes "
  "deterministic and cannot test workload seasonality. (viii) Cooling and PUE "
  "are excluded by scope. (ix) The linear power model is calibrated on "
  "u ≈ 0.1–0.7 and omits memory/IO effects. (x) The learned-policy evidence in "
  "Appendices A–B belongs to archived v1 and is not a v2 result.")

# ---------------- VIII. FUTURE WORK ----------------
EMIT_CONTENT = True
body_after_results = doc.add_section(WD_SECTION.CONTINUOUS)
set_cols(body_after_results, 2)
H1("VIII. FURTHER TOPICS OF CONSIDERATION")
P("(1) Batch-safe constrained control is first priority, and Sec. VII shows it "
  "cannot be reached by more optimizer sweeping or more compute alone: enforce "
  "terminal completion/deadline feasibility with a non-MPC joint feasibility/"
  "action-projection layer that maps every learned action onto the nearest "
  "feasible one with EXACT 0–100% drain reachability, rather than relying on a "
  "sigmoid decode whose ±3 logit bound caps reachable drain at 95.257% and "
  "whose open-interval range cannot reach exact full drain at any finite bound "
  "(Sec. VII-G) or on a penalty-mediated floor that Sec. VI and Sec. VII both "
  "show can still fail on the majority of seeds. (2) Improve spatial "
  "policy optimization, especially for Global where PPO captures only 5.47–"
  "7.55% of QP savings, and diagnose the asymmetric US transfer. (3) Add "
  "ramp-aware training or constraints because joint PPO worsens rare maximum "
  "three-hour ramps, while preserving the observed p95 improvement. (4) Add "
  "the secondary demand charge to a true multi-objective/Pareto study because "
  "joint PPO raises the reference bill 5.3–8.1%. (5) Bound Global routing with "
  "latency, residency, network capacity, and movement costs. (6) Add a causal "
  "receding-horizon MPC comparator — excluded from both the v2 and v3 "
  "protocols to date and not claimed anywhere in this thesis. (7) Test CAISO "
  "five-minute RTM, other energy months/years, and other workload traces. "
  "(8) Add carbon intensity, storage, and demand-response participation only "
  "after the core control problem is stable.")

# ---------------- IX. CONCLUSION ----------------
H1("IX. CONCLUSION")
P("This thesis builds a reproducible chain from measured Google service/batch "
  "usage and PowerData2019 calibration to one real May-2025 CAISO energy "
  "archetype, equal 100 MW proxy DCs, a joint PPO controller, and a frozen "
  "two-fold held-out evaluation. The system exposes substantial clairvoyant "
  "spatial headroom (7.2–16.4%) and modest incremental temporal headroom "
  "(0.6–2.3%). Learned performance is much narrower. Global spatial PPO saves "
  "0.90% and 1.19% on the two held-out folds with positive optimizer CIs and "
  "complete service, but captures only 5.47–7.55% of QP savings. US spatial "
  "does not establish savings. Joint batch PPO fails the frozen headline: only "
  "9/40 seeds meet the completion floor, gains are fold-dependent, and rare "
  "maximum ramps and the secondary demand-charge reference worsen. A post-hoc "
  "exploratory v3 study (Sec. VII) adds the missing episode/deadline context, "
  "but a post-run boundary audit shows the state repair remains incomplete; it "
  "re-screens optimizer and reward-weight hyperparameters and trains "
  "at up to 2× the frozen budget under a stricter pre-registered safety-first "
  "gate; it still fails — the recovered US configuration is economically "
  "indistinguishable from Status Quo and the recovered Global configuration is "
  "only 1/10-seed safe despite a real positive economic effect — so the joint "
  "failure is not an artifact of the omitted state or of training budget. The "
  "defensible contribution is therefore the real, audited experimental system; "
  "the modest optimistic Global spatial result; and the negative evidence, now "
  "twice obtained under two different state representations and training "
  "budgets, that unconstrained PPO does not solve reliable joint spatio-"
  "temporal shaping. The next step is not a stronger claim but safer batch "
  "control via action projection and better capture of demonstrable spatial "
  "opportunity.")

# ---------------- REFERENCES ----------------
H1("REFERENCES")
refs = [
 "M. Tirmazi, A. Barker, N. Deng, M. E. Haque, Z. G. Qin, S. Hand, M. Harchol-Balter, and J. Wilkes, “Borg: the Next Generation,” in Proc. EuroSys, 2020.",
 "V. Sakalkar, V. Kontorinis, D. Landhuis, S. Li, D. De Ronde, T. Blooming, A. Ramesh, J. Kennedy, C. Malone, J. Clidaras, and P. Ranganathan, “Data Center Power Oversubscription with a Medium Voltage Power Plane and Priority-Aware Capping,” in Proc. ASPLOS, 2020; companion dataset documentation: https://github.com/google/cluster-data/blob/master/PowerData2019.md.",
 "A. Radovanović, R. Koningstein, I. Schneider, B. Chen, A. Duarte, B. Roy, D. Xiao, M. Haridasan, P. Hung, N. Care, S. Talukdar, E. Mullen, K. Smith, M. Cottman, and W. Cirne, “Carbon-Aware Computing for Datacenters,” IEEE Trans. Power Systems, vol. 38, no. 2, pp. 1270–1280, Mar. 2023, doi: 10.1109/TPWRS.2022.3173250 (arXiv:2106.11750).",
 "L. Grange, G. Da Costa, and P. Stolf, “Green IT scheduling for data center powered with renewable energy,” Future Generation Computer Systems, vol. 86, pp. 99–120, 2018.",
 "G. Da Costa, L. Grange, and I. De Courchelle, “Modeling and generating large-scale Google-like workload,” in Proc. Int. Green and Sustainable Computing Conf. (IGSC), 2016.",
 "D. Zhao, J.-t. Zhou, and K. Li, “CFWS: DRL-Based Framework for Energy Cost and Carbon Footprint Optimization in Cloud Data Centers,” IEEE Trans. Sustainable Computing, vol. 10, no. 1, pp. 95–107, Jan./Feb. 2025, doi: 10.1109/TSUSC.2024.3391791.",
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
 "Z. Liu, M. Lin, A. Wierman, S. H. Low, and L. L. H. Andrew, “Greening Geographical Load Balancing,” in Proc. ACM SIGMETRICS, pp. 233–244, 2011, doi: 10.1145/1993744.1993767.",
 "A. Naug, A. Guillen, R. Luna, V. Gundecha, D. Rengarajan, S. Ghorbanpour, S. Mousavi, A. Ramesh Babu, D. Markovikj, L. D. Kashyap, and S. Sarkar, “SustainDC: Benchmarking for Sustainable Data Center Control,” in Advances in Neural Information Processing Systems (NeurIPS), 2024.",
 "Hewlett Packard Enterprise, “SustainCluster: A high-fidelity, open-source Gymnasium environment for benchmarking multi-objective, sustainable workload scheduling across geo-distributed data centers,” GitHub repository, MIT License, https://github.com/HewlettPackard/sustain-cluster (accessed Aug. 2026).",
 "International Energy Agency, “Energy and AI,” IEA, Paris, Jan. 2025. [Online]. Available: https://www.iea.org/reports/energy-and-ai.",
 "J. McLaren et al., “Identifying Potential Markets for Behind-the-Meter Battery Energy Storage: A Survey of U.S. Demand Charge Rates,” NREL/TP-6A20-64980, National Renewable Energy Laboratory, 2015.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    run = p.add_run(f"[{i}] {r}")
    run.font.size = Pt(8)
    p.paragraph_format.space_after = Pt(2)

doc.save(OUT)
print(f"Wrote {OUT}")
