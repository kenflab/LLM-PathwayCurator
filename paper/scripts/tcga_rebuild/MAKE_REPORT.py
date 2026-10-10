# ruff: noqa: E501
# Embedded HTML preserves the verified report layout.
#!/usr/bin/env python3
"""Create the corrected OV review page, separating genotype and analysis denominators."""

import argparse
from html import escape
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    root = ap.parse_args().run_dir
    summary = pd.read_csv(root / "cohort_summary.tsv", sep="\t")
    stages = pd.read_csv(root / "OV_denominators.tsv", sep="\t")
    rows = []
    for r in summary.itertuples(index=False):
        n = str(int(r.n_q_le_0_05)) if pd.notna(r.n_q_le_0_05) else "計算不可"
        link = (
            f'<a href="source_reports/{escape(r.cancer)}/report.html">開く</a>'
            if r.fit_eligible
            else "—"
        )
        rows.append(
            f"<tr><th>{r.cancer}</th><td>{r.n_mut}</td><td>{r.n_wt}</td><td>{r.n_unknown}</td><td>{n}</td><td>{link}</td></tr>"
        )
    stage_rows = []
    labels = [
        "MC3測定情報を確認（RNA品質除外前）",
        "RNA品質・重複列除外後",
        "GDCとの既知の判定不一致を除外後",
    ]
    for label, r in zip(labels, stages.itertuples(index=False), strict=True):
        stage_rows.append(
            f"<tr><th>{label}</th><td>{r.MUT}</td><td>{r.MC3_call_negative}</td><td>{r.excluded_unknown}</td></tr>"
        )
    html = """<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OV修正 — LLM-PathwayCurator</title><style>
body{font:16px/1.8 system-ui,sans-serif;background:#f3f6f8;color:#243649;margin:0}main{max-width:1040px;margin:auto;padding:36px 26px;background:white}h1{font-size:30px;line-height:1.4}h2{font-size:22px;margin-top:32px}a{color:#145e87}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #dae1e8;text-align:right}th:first-child{text-align:left}thead{background:#edf2f6}.note{background:#fff4df;border-left:4px solid #bd7828;padding:15px 20px}.small{font-size:13px;color:#566a78}img{width:100%;height:auto}code{overflow-wrap:anywhere;font-size:13px}li{margin-bottom:7px}</style><main>
<p class="small">LLM-PATHWAYCURATOR · OVFIX1 · 2026-10-09 (America/Chicago)</p>
<h1>OVの過剰除外を修正しました</h1>
<div class="note"><strong>旧「MUT 3例／WT 1例」は、OVを代表する群分けとして使用しないでください。</strong><br>WGAサンプルを一律に除外した結果でした。修正版は文献に基づくOVのwga-only採用例外を実装し、現在のGDCと判定が不一致の比較候補も除外しています。</div>
<h2>OVの分母を分けて確認</h2>
<p>今回の発現入力に含まれるOVは303例です。TCGA-OVプロジェクト全体や原著の解析対象数とは異なります。</p>
<table><thead><tr><th>処理段階</th><th>TP53 MUT</th><th>MC3 call-negative*</th><th>除外・不明</th></tr></thead><tbody>STAGE_ROWS</tbody></table>
<p class="small">* 宣言した方針で採用するMC3 TP53変異がない群。生物学的WTを確認した群ではありません。RNA品質が不適格なMUT例も、変異を持つという記録は台帳に保持します。</p>
<p>GDCには、比較候補の <code>TCGA-24-1430</code> にTP53 missense変異、<code>TCGA-61-1910</code> にTP53 frameshift変異が記録されていました。この2例を比較群から除外しました。照合はcase単位であり、MC3とGDCを混ぜてMUTに再分類していません。GDCで記録が見つからないことをWTの証明には使いません。</p>
<p><a href="OV_sample_changes.tsv">OV全303例の旧→新判定</a> · <a href="OV_GDC_discordance_evidence.tsv">GDC不一致の変異記録</a> · <a href="OV_denominators.tsv">段階別集計</a></p>
<h2>修正版での経路解析</h2>
<table><thead><tr><th>コホート</th><th>MUT</th><th>MC3 call-negative</th><th>除外・不明</th><th>q≤0.05 / 50</th><th>sourceレポート</th></tr></thead><tbody>RESULT_ROWS</tbody></table>
<p>OVの比較群は7例、SKCMのMUT群は9例と少数です。未調整の診断的比較であり、OVの結果を強固なTP53効果の証明には使いません。他の6コホートのサンプル群分けは前回と一致することを照合しました。</p>
<img src="figures/cohorts_and_enrichment.png" alt="Corrected cohort composition and enrichment counts">
<p><a href="cohort_summary.tsv">全コホート集計</a> · <a href="sample_ledger.tsv">全サンプル台帳</a> · <a href="hallmark_results.tsv">全50経路の統計結果</a></p>
<img src="figures/hallmark_NES.png" alt="All Hallmark normalized enrichment scores">
<p><a href="figures/cohorts_and_enrichment.pdf">集計図 PDF</a> · <a href="figures/hallmark_NES.pdf">全経路図 PDF</a></p>
<h2>変更したルールと根拠</h2>
<ul><li>OVのみ、FILTERが正確に <code>PASS</code> または <code>wga</code> のTP53変異を採用します。<code>wga,oxog</code> など他のフィルタが併記された変異は復活させません。元のFILTER値は保存します。</li>
<li>OVではWGA由来DNAを一律除外しません。測定ペア、RNA品質、混在フラグ等の確認は継続します。他の6コホートのnative-DNA/PASS方針は変更していません。</li>
<li>GDCの公開変異記録を固定した入力として保存し、OVの既知のMC3陰性／GDC陽性不一致を比較群から除外します。この追加照合はOVのみです。</li>
<li>現在の公開コミット <code>764b18ec0140d9c878bf7866f129e6513ac751d4</code> の群割当関数・ランキング・fgsea・sourceレポートを使用します。コホート別の変異採用と不一致除外は、このZIPのRUN.pyが担います。公開リポジトリ自体は今回変更していません。</li>
<li>MSigDB 2026.1.Hs Hallmarkを使用し、集合スナップショットの一致を確認します。モデル呼び出し0回。生物学的正確性やLLM性能の検証を意味しません。</li></ul>
<p>TCGA原著の高異型度漿液性卵巣癌ではTP53変異96%と報告されています。ただし検体、検出方法、データ版が異なるため、今回の群数を96%に合わせて調整していません。GDCの画像の値と、この303例からの解析対象数も直接比較できません。</p>
<ul><li><a href="https://gdc.cancer.gov/about-data/publications/ov_2011">TCGA ovarian carcinoma, Nature 2011</a></li>
<li><a href="https://doi.org/10.1016/j.cell.2018.02.060">Bailey et al., Cell 2018</a> — OV/LAMLのwga-only変異の採用例外。</li>
<li><a href="https://gdc.cancer.gov/about-data/publications/mc3-2017">MC3公開入力</a></li>
<li><a href="https://portal.gdc.cancer.gov/genes/ENSG00000141510">GDC TP53</a> — 動的な画面の割合はこの再解析の分母に使用しません。</li></ul>
<p><a href="OV_audit.json">OV監査記録</a> · <a href="result_verification.json">結果・レポート照合</a> · <a href="RUN_STATUS.json">実行状態</a> · <a href="ANALYSIS_POLICY.json">解析条件</a> · <a href="INPUT_MANIFEST.json">入力来歴</a></p>
</main></html>"""
    (root / "START_HERE.html").write_text(
        html.replace("STAGE_ROWS", "\n".join(stage_rows)).replace("RESULT_ROWS", "\n".join(rows)),
        encoding="utf-8",
    )
    print((root / "START_HERE.html").resolve())


if __name__ == "__main__":
    main()
