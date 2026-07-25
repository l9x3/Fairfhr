import time
from config import ensure_dirs, DATA_PATH, OUTPUT_DIR
import prep, pipeline, global_shap, analysis, comparison, crossdataset, figures, report


def main():
    ensure_dirs()
    t0 = time.time()
    print(f"[data] loading {DATA_PATH}")
    work, feat = prep.get_working_data()
    print(f"[data] working subset {work.shape}, {work['Subject'].nunique()} subjects")

    print("\n== cross-validation pipeline ==")
    res = pipeline.run(work, feat)

    print("\n== global SHAP + k-selection ==")
    art = global_shap.build_global(work, feat)
    art = global_shap.k_selection(work, feat, art)

    print("\n== tables + statistical tests ==")
    analysis.make_tables(res["oof"], res["fold_metrics"], work, res["hp_search"])

    print("\n== comparison methods ==")
    comparison.make_tables(work, feat, res["oof"])

    print("\n== cross-dataset evaluation ==")
    crossdataset.make_tables(work, feat, res["oof"])

    print("\n== figures + report ==")
    figures.make_figures(work, feat, art)
    out = report.build()

    print(f"\nDONE in {time.time()-t0:.0f}s. Outputs in {OUTPUT_DIR}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
