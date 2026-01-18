# LLM-PathwayCurator report

## Sample Card
- disease: HNSC
- tissue: oral_cavity
- perturbation: TP53_mut_vs_wt
- comparison: tumor
- notes: Demo sample card for LLM-PathwayCurator.

## Decisions (PASS/ABSTAIN/FAIL): 0/3/0
PASS requires evidence-link integrity and stability; otherwise the system ABSTAINS or FAILS.

## Risk–Coverage summary

- coverage_pass_total: 0.000
- risk_fail_given_decided: nan
- risk_fail_total: 0.000
- score_col_for_curve: term_survival_agg

## Reason summary

### ABSTAIN reasons
| abstain_reason   |   n |
|:-----------------|----:|
| missing_survival |   3 |

## PASS (top)

| claim_id   | entity   | direction   | context_keys   | term_uid   | source   | term_name   | module_id   | gene_ids   | term_ids   | gene_set_hash   | context_score   | claim_json   | status   | link_ok   | stability_ok   | contradiction_ok   | stress_ok   | abstain_reason   | fail_reason   | audit_notes   | term_survival_agg   |
|------------|----------|-------------|----------------|------------|----------|-------------|-------------|------------|------------|-----------------|-----------------|--------------|----------|-----------|----------------|--------------------|-------------|------------------|---------------|---------------|---------------------|

## ABSTAIN (top)

| claim_id       | entity              | direction   | module_id       | abstain_reason   | gene_set_hash   |   term_survival_agg |   context_score | audit_notes                                |
|:---------------|:--------------------|:------------|:----------------|:-----------------|:----------------|--------------------:|----------------:|:-------------------------------------------|
| c_84a71cc47d7b | p53 pathway         | up          | M0001__5dccc39b | missing_survival | 5fa369d2f7ae    |                 nan |               0 | term_survival missing for referenced terms |
| c_e9a164384210 | G2M checkpoint      | up          | M0002__ea9cd2d8 | missing_survival | cca2b2acae3e    |                 nan |               0 | term_survival missing for referenced terms |
| c_f47ee13eddd6 | DNA damage response | na          | M0001__5dccc39b | missing_survival | 792930786779    |                 nan |               0 | term_survival missing for referenced terms |

## FAIL (top)

| claim_id   | entity   | direction   | context_keys   | term_uid   | source   | term_name   | module_id   | gene_ids   | term_ids   | gene_set_hash   | context_score   | claim_json   | status   | link_ok   | stability_ok   | contradiction_ok   | stress_ok   | abstain_reason   | fail_reason   | audit_notes   | term_survival_agg   |
|------------|----------|-------------|----------------|------------|----------|-------------|-------------|------------|------------|-----------------|-----------------|--------------|----------|-----------|----------------|--------------------|-------------|------------------|---------------|---------------|---------------------|

## Audit log (top)

| claim_id       | entity              | direction   | context_keys                           | term_uid                      | source    | term_name           | module_id       | gene_ids               | term_ids                      | gene_set_hash   |   context_score | claim_json                                                                                                                                                                                                                                                                                          | status   | link_ok   | stability_ok   | contradiction_ok   | stress_ok   | abstain_reason   | fail_reason   | audit_notes                                | term_survival_agg   |
|:---------------|:--------------------|:------------|:---------------------------------------|:------------------------------|:----------|:--------------------|:----------------|:-----------------------|:------------------------------|:----------------|----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|:----------|:---------------|:-------------------|:------------|:-----------------|:--------------|:-------------------------------------------|:--------------------|
| c_f47ee13eddd6 | DNA damage response | na          | disease,tissue,perturbation,comparison | metascape:GO:0006977          | metascape | DNA damage response | M0001__5dccc39b | ATM,ATR,BRCA1,TP53     | metascape:GO:0006977          | 792930786779    |               0 | {"claim_id":"c_f47ee13eddd6","entity":"DNA damage response","direction":"na","context_keys":["disease","tissue","perturbation","comparison"],"evidence_ref":{"module_id":"M0001__5dccc39b","gene_ids":["ATM","ATR","BRCA1","TP53"],"term_ids":["metascape:GO:0006977"],"gene_set_hash":""}}         | ABSTAIN  | True      | False          | True               | True        | missing_survival |               | term_survival missing for referenced terms |                     |
| c_84a71cc47d7b | p53 pathway         | up          | disease,tissue,perturbation,comparison | fgsea:HALLMARK_P53_PATHWAY    | fgsea     | p53 pathway         | M0001__5dccc39b | BAX,CDKN1A,MDM2,TP53   | fgsea:HALLMARK_P53_PATHWAY    | 5fa369d2f7ae    |               0 | {"claim_id":"c_84a71cc47d7b","entity":"p53 pathway","direction":"up","context_keys":["disease","tissue","perturbation","comparison"],"evidence_ref":{"module_id":"M0001__5dccc39b","gene_ids":["BAX","CDKN1A","MDM2","TP53"],"term_ids":["fgsea:HALLMARK_P53_PATHWAY"],"gene_set_hash":""}}         | ABSTAIN  | True      | False          | True               | True        | missing_survival |               | term_survival missing for referenced terms |                     |
| c_e9a164384210 | G2M checkpoint      | up          | disease,tissue,perturbation,comparison | fgsea:HALLMARK_G2M_CHECKPOINT | fgsea     | G2M checkpoint      | M0002__ea9cd2d8 | CCNB1,CDK1,MKI67,TOP2A | fgsea:HALLMARK_G2M_CHECKPOINT | cca2b2acae3e    |               0 | {"claim_id":"c_e9a164384210","entity":"G2M checkpoint","direction":"up","context_keys":["disease","tissue","perturbation","comparison"],"evidence_ref":{"module_id":"M0002__ea9cd2d8","gene_ids":["CCNB1","CDK1","MKI67","TOP2A"],"term_ids":["fgsea:HALLMARK_G2M_CHECKPOINT"],"gene_set_hash":""}} | ABSTAIN  | True      | False          | True               | True        | missing_survival |               | term_survival missing for referenced terms |                     |

## Reproducible artifacts

- audit_log.tsv: audit_log.tsv
- distilled.tsv: distilled.tsv
