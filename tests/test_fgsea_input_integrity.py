import pandas as pd
import pytest

from llm_pathway_curator.adapters.fgsea import fgsea_to_evidence_table


def result(**extra):
    return pd.DataFrame(
        {"pathway": ["P"], "NES": [1.8], "padj": [0.02], "leadingEdge": ["TP53,CDKN1A"], **extra}
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_q_columns_never_depend_on_column_order(reverse):
    df = result(FDR=[0.8])
    if reverse:
        df = df[df.columns[::-1]]
    with pytest.raises(ValueError, match="ambiguous column aliases"):
        fgsea_to_evidence_table(df)


def test_duplicate_column_labels_are_rejected():
    df = result()
    df = pd.concat([df, df[["padj"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate input column"):
        fgsea_to_evidence_table(df)


@pytest.mark.parametrize("q", [-0.01, 1.01, float("inf"), "not-a-q-value"])
def test_invalid_adjusted_p_values_cannot_become_missing_rows(q):
    with pytest.raises(ValueError, match=r"padj must be missing or finite in \[0, 1\]"):
        fgsea_to_evidence_table(result(padj=[q]))


@pytest.mark.parametrize("q", [0.0, 0.05, 1.0])
def test_valid_q_values_and_declared_namespace_are_preserved(q):
    ev = fgsea_to_evidence_table(result(padj=[q], gene_id_type=["symbol"]))
    assert ev.qval.iloc[0] == q
    assert ev.gene_id_type.iloc[0] == "symbol"


def test_real_missing_q_and_numeric_gene_ids_remain_supported():
    assert fgsea_to_evidence_table(result(padj=[None])).empty
    ev = fgsea_to_evidence_table(result(leadingEdge=["7157,1026"], gene_id_type=["entrez"]))
    assert ev.evidence_genes.iloc[0] == ["7157", "1026"]
    assert ev.gene_id_type.iloc[0] == "entrez"
