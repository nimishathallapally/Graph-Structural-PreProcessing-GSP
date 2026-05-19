"""Generate LaTeX longtables for the complete sweep results."""
import json
import os

BASE = r"e:\gsp\recosys\output"

DATASETS = [
    ("ML-1M",  "sweep_ml1m"),
    ("Yelp",   "sweep_yelp"),
    ("ML-25M", "sweep_ml25m_ordered"),
]

MODES      = ["forman_ricci", "cosine"]
FRACS      = ["025", "05", "075", "10"]
MSS        = ["1", "3", "5"]
FRAC_LABEL = {"025": "0.25", "05": "0.50", "075": "0.75", "10": "1.00"}
MODELS     = ["lightgcn", "gcn", "graphsage", "gat"]
MODEL_LABEL= {"lightgcn": "LightGCN", "gcn": "GCN",
               "graphsage": "GraphSAGE", "gat": "GAT"}


def load(dataset_dir, mode, frac, ms):
    folder = f"{mode}_frac{frac}_ms{ms}"
    path   = os.path.join(BASE, dataset_dir, folder, "full_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def gsp_stats(data):
    gsp = data.get("gsp", {})
    nodes = gsp.get("compression_ratio_pct", 0.0)
    edges = gsp.get("bipartite_edge_reduction_pct", 0.0)
    prepro= gsp.get("gsp_preprocessing_time_s", 0.0)
    return f"{nodes:.2f}", f"{edges:.2f}", f"{prepro:.1f}"


def get_metric(data, model, run_type, metric):
    metrics_by = {(m.get("model"), m.get("run_type")): m
                  for m in data.get("metrics", [])}
    entry = metrics_by.get((model, run_type), {})
    val = entry.get(metric, float("nan"))
    if val != val:  # nan check
        return "---"
    return f"{val:.4f}"


def build_longtable(dataset_name, dataset_dir, mode):
    mode_label = "Forman-Ricci" if mode == "forman_ricci" else "Cosine Similarity"
    tab_label  = f"tab:{dataset_name.lower().replace('-','').replace(' ','_')}_{mode}_sweep"
    caption    = (f"Complete sweep results for {dataset_name} with "
                  f"{mode_label} curvature. "
                  r"NDCG@10 and Recall@10 shown for baseline (Base) and "
                  r"GSP-projected (GSP) modes.")

    lines = []
    lines.append(r"{\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{longtable}{cccccl rrrr}")
    lines.append(rf"\caption{{{caption}}} \label{{{tab_label}}} \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Frac} & \multirow{2}{*}{MS} & "
                 r"\multirow{2}{*}{Nodes\%} & \multirow{2}{*}{Edges\%} & "
                 r"\multirow{2}{*}{Prepro (s)} & \multirow{2}{*}{Model} & "
                 r"\multicolumn{2}{c}{NDCG@10} & \multicolumn{2}{c}{Recall@10} \\")
    lines.append(r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
    lines.append(r" & & & & & & Base & GSP & Base & GSP \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{10}{c}{\tablename~\thetable{} -- \textit{continued}} \\")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Frac} & \multirow{2}{*}{MS} & "
                 r"\multirow{2}{*}{Nodes\%} & \multirow{2}{*}{Edges\%} & "
                 r"\multirow{2}{*}{Prepro (s)} & \multirow{2}{*}{Model} & "
                 r"\multicolumn{2}{c}{NDCG@10} & \multicolumn{2}{c}{Recall@10} \\")
    lines.append(r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}")
    lines.append(r" & & & & & & Base & GSP & Base & GSP \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule \multicolumn{10}{r}{\textit{continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    any_data = False
    for frac in FRACS:
        for ms in MSS:
            data = load(dataset_dir, mode, frac, ms)
            if data is None:
                continue
            any_data = True
            nodes_s, edges_s, prepro_s = gsp_stats(data)
            frac_lbl = FRAC_LABEL[frac]
            for i, model in enumerate(MODELS):
                ndcg_b = get_metric(data, model, "baseline",     "NDCG@10")
                ndcg_g = get_metric(data, model, "gsp_projected","NDCG@10")
                rec_b  = get_metric(data, model, "baseline",     "Recall@10")
                rec_g  = get_metric(data, model, "gsp_projected","Recall@10")
                if i == 0:
                    row = (rf"\multirow{{4}}{{*}}{{{frac_lbl}}} & "
                           rf"\multirow{{4}}{{*}}{{{ms}}} & "
                           rf"\multirow{{4}}{{*}}{{{nodes_s}}} & "
                           rf"\multirow{{4}}{{*}}{{{edges_s}}} & "
                           rf"\multirow{{4}}{{*}}{{{prepro_s}}} & "
                           rf"{MODEL_LABEL[model]} & "
                           rf"{ndcg_b} & {ndcg_g} & {rec_b} & {rec_g} \\")
                else:
                    row = (rf" & & & & & {MODEL_LABEL[model]} & "
                           rf"{ndcg_b} & {ndcg_g} & {rec_b} & {rec_g} \\")
                lines.append(row)
            lines.append(r"\midrule")

    if not any_data:
        return None

    lines.append(r"\end{longtable}")
    lines.append(r"}")  # close \footnotesize
    return "\n".join(lines)


def build_ml25m_table():
    """ML-25M has only forman_ricci frac10 ms1."""
    data = load("sweep_ml25m_ordered", "forman_ricci", "10", "1")
    if data is None:
        return None
    nodes_s, edges_s, prepro_s = gsp_stats(data)

    lines = []
    lines.append(r"{\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{ML-25M sweep results: Forman-Ricci, frac=1.00, ms=1. "
                 r"NDCG@10 and Recall@10 for baseline (Base) and GSP-projected (GSP). "
                 r"GAT did not converge on this scale (NaN values omitted).}")
    lines.append(r"\label{tab:ml25m_sweep}")
    lines.append(r"\begin{tabular}{l rrrr}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Model} & \multicolumn{2}{c}{NDCG@10} & \multicolumn{2}{c}{Recall@10} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    lines.append(r" & Base & GSP & Base & GSP \\")
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{Nodes reduced: {nodes_s}\%, "
                 rf"Edges reduced: {edges_s}\%, Prepro: {prepro_s}\,s}}}} \\")
    lines.append(r"\midrule")
    for model in MODELS:
        ndcg_b = get_metric(data, model, "baseline",      "NDCG@10")
        ndcg_g = get_metric(data, model, "gsp_projected", "NDCG@10")
        rec_b  = get_metric(data, model, "baseline",      "Recall@10")
        rec_g  = get_metric(data, model, "gsp_projected", "Recall@10")
        if ndcg_b == "---" and ndcg_g == "---":
            continue
        lines.append(rf"{MODEL_LABEL[model]} & {ndcg_b} & {ndcg_g} & {rec_b} & {rec_g} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"}")
    return "\n".join(lines)


def main():
    output_parts = []

    output_parts.append(r"""\section{Complete Sweep Results}\label{sec:complete_sweep}

Tables~\ref{tab:ml1m_forman_ricci_sweep}--\ref{tab:ml25m_sweep} present the complete NDCG@10 and Recall@10 results for every hyperparameter configuration tested across all three datasets. Each configuration is identified by the curvature fraction (\textit{frac}) and minimum cluster size (\textit{MS}). \textit{Nodes\%} and \textit{Edges\%} report the percentage reduction in bipartite graph nodes and edges after GSP processing; \textit{Prepro~(s)} is the wall-clock preprocessing time in seconds. Baseline (Base) values are from training each GNN on the full uncompressed graph; GSP values are from training on the projected compressed graph.
""")

    MODE_LABELS = {"forman_ricci": "Forman-Ricci", "cosine": "Cosine Similarity"}

    # ML-1M tables
    output_parts.append(r"\subsection*{MovieLens-1M Results}")
    for mode in MODES:
        output_parts.append(rf"\subsubsection*{{Curvature Mode: {MODE_LABELS[mode]}}}")
        tbl = build_longtable("ML-1M", "sweep_ml1m", mode)
        if tbl:
            output_parts.append(tbl)
            output_parts.append("")

    # Yelp tables
    output_parts.append(r"\subsection*{Yelp Results}")
    for mode in MODES:
        output_parts.append(rf"\subsubsection*{{Curvature Mode: {MODE_LABELS[mode]}}}")
        tbl = build_longtable("Yelp", "sweep_yelp", mode)
        if tbl:
            output_parts.append(tbl)
            output_parts.append("")

    # ML-25M
    output_parts.append(r"\subsection*{MovieLens-25M Results (Single Configuration)}")
    # ML-25M
    subsection_ml25m = r"\subsection*{MovieLens-25M -- Forman-Ricci (frac=1.00, ms=1)}"
    output_parts.append(subsection_ml25m)
    tbl = build_ml25m_table()
    if tbl:
        output_parts.append(tbl)
        output_parts.append("")

    result = "\n".join(output_parts)

    out_path = r"e:\gsp\recosys\scripts\_tables_output.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Written {len(result)} chars to {out_path}")
    print(result[:2000])


if __name__ == "__main__":
    main()
