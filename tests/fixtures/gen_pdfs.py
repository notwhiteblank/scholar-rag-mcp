from __future__ import annotations

import html
import sys
from pathlib import Path

from fpdf import FPDF

FIXTURES_DIR = Path(__file__).resolve().parent


def _escaped(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2014", "-").replace("\u2013", "-")
    return html.unescape(text)


PAPER_A_BLOCKS: list[tuple[str, str]] = [
    (
        "h1",
        "Improved Crop Yield Forecasting with Multi-Sensor Time Series",
    ),
    ("p", "First Author, Second Author, and Third Author"),
    ("p", "Institute of Applied Sciences, University of Example"),
    ("h2", "Abstract"),
    (
        "p",
        "Accurate crop yield prediction is essential for food security planning and agricultural policy. "
        "We present a forecasting pipeline that fuses satellite vegetation indices with local weather "
        "records. On a multi-year benchmark the proposed model reduces mean absolute error by 18 percent "
        "relative to a strong baseline while remaining interpretable.",
    ),
    ("h2", "Introduction"),
    (
        "p",
        "Yield forecasting has long relied on statistical models fitted to historical farm surveys. "
        "Recent access to high-resolution remote sensing opened new opportunities for data-driven "
        "approaches. Still, most published systems are evaluated on single regions and seasons.",
    ),
    ("p", "In this work we study whether geographic and seasonal variation can be handled by a "
        "single unified model. Our results indicate that careful alignment of weather and image "
        "features matters more than model complexity."),
    ("h3", "Contributions"),
    ("p", "We release an open preprocessing toolkit and report ablations on three public datasets. "
        "The main contribution is a normalization scheme that stabilizes training across regions."),
    ("h2", "Results"),
    ("p", "Table 1 summarizes the held-out error of all compared models. The proposed model achieves "
        "the lowest error on every region while its runtime stays well below the seasonal deadline."),
    ("h3", "Comparison with Baselines"),
    ("p", "When measured against ridge regression and gradient boosting, the fusion model wins in "
        "eight out of nine experimental cells. The remaining cell is tied within one standard error."),
    ("h3", "Seasonal Robustness"),
    ("p", "Splitting the test period by growing season shows no performance cliff on dry years. "
        "This suggests the weather normalization absorbs most environmental drift."),
    ("h2", "Discussion"),
    ("p", "Our findings demonstrate that multi-sensor fusion is a practical route to robust forecasts. "
        "The failure of the strong baseline in early season is consistent with its limited view of soil "
        "moisture dynamics."),
    ("p", "We anticipate that extending the pipeline to include economic indicators will further close "
        "the gap between forecasts and administrative decisions."),
    ("h2", "Methods"),
    ("p", "We used publicly available satellite imagery resampled to a uniform grid. Weather variables "
        "were sourced from a global reanalysis product and interpolated to station locations."),
    ("h3", "Model Architecture"),
    ("p", "The model combines a two-layer temporal encoder with a lightweight attention module. "
        "All layers were trained with standard gradient descent and early stopping on a validation fold."),
    ("h3", "Evaluation Protocol"),
    ("p", "We performed five-fold cross-validation stratified by region. Metrics are reported as mean "
        "absolute error in tonnes per hectare together with the standard deviation over folds."),
    ("h2", "References"),
    ("p", "1. Smith J, Doe A. A survey of statistical yield models. Journal of Agricultural Computing. 2019;12(4):101-118."),
    ("p", "2. Lee K, Chen R. Remote sensing for crop monitoring. Sensors and Systems. 2020;33(1):55-73."),
    ("p", "3. Wang X, Patel S. Weather normalization in forecasting pipelines. Data Sciences Review. 2022;27(2):200-214."),
]

PAPER_B_BLOCKS: list[tuple[str, str]] = [
    ("h1", "Graph Neural Networks for Wireless Network Routing: A Review"),
    ("p", "A Review Author, Collaborator Author, and Senior Author"),
    ("p", "Department of Computer Science, Example University"),
    ("p", "Corresponding author: a.review@example.org"),
    ("h2", "Summary"),
    (
        "p",
        "Graph neural networks offer a unified way to model communication networks whose topology "
        "changes over time. This review covers the design space, highlights open evaluation gaps, "
        "and sketches directions for future work. We survey more than sixty papers published "
        "between 2018 and 2025.",
    ),
    ("h2", "Background"),
    (
        "p",
        "Classical routing protocols react to congestion with hand-tuned heuristics. As network "
        "topologies become denser, learning-based policies trained on simulated environments gain "
        "attention because they can adapt to traffic without explicit modeling.",
    ),
    ("h3", "Message Passing Schemes"),
    ("p", "Most proposals cast the network as a graph whose edges carry delay and capacity. "
        "Message passing over this graph lets a policy combine local congestion signals into "
        "globally sensible forwarding decisions."),
    ("h3", "Training Regimes"),
    ("p", "A recurring distinction is between supervised imitation of expert routes and pure "
        "reinforcement learning. The two regimes differ sharply in sample efficiency and in the "
        "stability of the learned policy under distribution shift."),
    ("h2", "Open Challenges"),
    ("p", "Evaluation remains fragmented: few studies share a common simulator, traffic model, or "
        "metric definition. This makes head-to-head comparison difficult and slows adoption in "
        "operational settings."),
    ("h3", "Scalability"),
    ("p", "Graph policies trained on small topologies often fail to transfer to larger or denser "
        "networks. Scaling laws for such policies are largely unexplored."),
    ("h3", "Robustness"),
    ("p", "We found that small perturbations of link weights can flip routing decisions. Robustness "
        "against measurement noise should therefore become a standard evaluation axis."),
    ("h2", "Conclusion"),
    ("p", "We summarize the emerging consensus that graph-based routing is promising but its "
        "evaluation science lags behind algorithm design. We propose a shared benchmark and a "
        "robustness checklist for future studies."),
    ("h2", "Bibliography"),
    ("p", "[1] Miller T. Routing in dynamic networks. Network Systems. 2018;9(2):15-30."),
    ("p", "[2] Gomez F, Zhang Y. Learning to route with graph embeddings. IT Transactions. 2021;14(3):88-99."),
    ("p", "[3] Rossi M, Klein D. Simulating congestion events. Simulation Practice. 2023;19(1):7-21."),
]

PAPER_C_BLOCKS: list[tuple[str, str]] = [
    ("h1", "Prompt Ensembles Improve Few-Shot Text Classification Accuracy"),
    ("p", "C. Inventor and D. Builder"),
    ("p", "Center for Language Technology, Example Institute"),
    ("h2", "Abstract"),
    (
        "p",
        "Few-shot text classification is sensitive to the wording of prompts. We show that averaging "
        "predictions over a small ensemble of manually written prompts yields consistent gains while "
        "costing only a linear increase in inference time.",
    ),
    ("h2", "1. Introduction"),
    ("p", "Prompting a large language model is now the default strategy for low-resource tasks. "
        "The choice of prompt, however, remains brittle and is rarely documented in published "
        "results."),
    ("p", "We investigated whether ensembles of plausible prompts can reduce this brittleness "
        "without any additional training data."),
    ("h2", "2. Approach"),
    ("p", "For every example we generate one score vector per prompt and average the vectors before "
        "applying the decision threshold. No model weights are updated during the entire process."),
    ("h3", "Prompt Selection"),
    ("p", "Prompts were written independently by three analysts following a short style guide. "
        "The guide constrained prompt length and forbade mention of the test labels."),
    ("h3", "Aggregation"),
    ("p", "We experimented with arithmetic and geometric averaging and found the two almost "
        "indistinguishable on the development set."),
    ("h2", "3. Experiments"),
    ("p", "On six public benchmarks the ensemble improves mean accuracy by 4.1 points in the "
        "eight-shot setting. The gain is largest for tasks with high label ambiguity."),
    ("h2", "4. Discussion"),
    ("p", "We interpret the gain as variance reduction: single prompts act as strong prior views, "
        "and averaging stabilizes the decision boundary. Future work should study adaptive "
        "ensembles whose members are pruned per task."),
    ("h2", "5. Conclusion"),
    ("p", "Prompt ensembles are a simple, training-free remedy for prompt brittleness. We recommend "
        "them as a default reporting practice in few-shot research."),
    ("h2", "References"),
    ("p", "1. Brown A, White B. Large language models as few-shot learners. In Proceedings of the "
        "Conference on Big Science. 2020."),
    ("p", "2. Grey C. A taxonomy of prompting strategies. Language Notes. 2021;5(1):33-49."),
    ("p", "3. Hill D, Green E. Ensembling for robustness. Machine Learning Matters. 2022;8(2):12-20."),
]

PAPERS: dict[str, list[tuple[str, str]]] = {
    "paper_a": PAPER_A_BLOCKS,
    "paper_b": PAPER_B_BLOCKS,
    "paper_c": PAPER_C_BLOCKS,
}


def to_markdown(blocks: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for kind, text in blocks:
        if kind == "h1":
            lines.append(f"# {text}")
        elif kind == "h2":
            lines.append(f"## {text}")
        elif kind == "h3":
            lines.append(f"### {text}")
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def render_pdf(path: Path, blocks: list[tuple[str, str]]) -> None:
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 18, 20)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    for kind, raw in blocks:
        text = _escaped(raw)
        if kind in ("h1", "h2", "h3"):
            size = {"h1": 14, "h2": 12, "h3": 11}[kind]
            pdf.set_font("Helvetica", style="B", size=size)
        else:
            pdf.set_font("Helvetica", style="", size=10)
        pdf.multi_cell(0, 6, text)
        pdf.ln(2)
    path.write_bytes(pdf.output())


def generate_all(output_dir: Path | None = None) -> dict[str, Path]:
    target_dir = Path(output_dir) if output_dir is not None else FIXTURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, blocks in PAPERS.items():
        pdf_path = target_dir / f"{name}.pdf"
        render_pdf(pdf_path, blocks)
        written[name] = pdf_path
    return written


def main() -> None:
    written = generate_all()
    print(f"Generated {len(written)} PDFs in {FIXTURES_DIR}:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
