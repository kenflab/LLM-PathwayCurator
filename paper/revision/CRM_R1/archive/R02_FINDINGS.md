# R02: figure provenance and external dataset feasibility

Reviewed on 2026-10-04 against main `7e8c3acd68a10c64e3b06775566c84c1a0c8ddf4`.
This is a historical-source audit and metadata check. It estimates no new
biological or semantic performance. Original P1-P5 results remain intact.

The supplied submission PDF and Document S1 were rendered and inspected.
The reviewed file hashes, 18 empirical panel routes, and 970 public source
hashes are recorded in `config/r02_reviewed_sources.json`. Visual correspondence
and matching stored sources do not recover the exact final assembly commands.
Any differing PDF copy makes the layout review stale.

## Findings affecting the submitted figures

| Submitted panel | Source issue | Revision action |
| --- | --- | --- |
| Fig. 2e/f | The submission uses bars for e and packed modules for f. FIGURE_MAP declares the opposite. Both use the stored hash-dependent utility. | Align map, plot, legend, Source Data and text with the final replacement figure. |
| Fig. S2b/c | Submitted b is outcomes and c is module statistics. The source PDF and CSV names put these under c and b, respectively. | Align the final panel labels and exports. |
| Fig. S2 GO/BRCA | The run metadata says aborted at distill, although audit/module artifacts and plotted counts remain. | Preserve the artifacts. Locate the successful original execution record or remove this example from performance claims. Do not silently rerun and relabel it. |
| Fig. S3a/b | Active LLM proposal/review and a raw context cache are recorded, but backend_identity is absent. Script 96 cannot certify the model name. | Keep this as a limited historical slice. A future experiment must capture model identity, prompts and raw responses. Do not manufacture missing historical metadata. |
| Fig. S4f | The submitted panel is direction-colored packed circles. The map/legend describes a bar plot. | Align the final panel content and legend. |
| Ranked Source Data CSVs | 100 cells in score_source contain `#NAME?`, rather than the source label stored in the ranked TSV. | Fix the new export pathway and compare it with the stored source. Do not interpret these label errors as changes in the numerical scores. |

149 run records were inspected: 148 have status ok and one has status aborted.
147 explicitly record proxy context review, one explicitly records LLM review,
and the aborted record lacks the completed claims metadata. All 7,400 proxy
artifact values across 148 non-LLM source bundles reproduce the historical hash.
This includes the retained artifacts beside the aborted record and does not
establish successful execution for that record. All 149 normalized evidence/card
hash pairs match their recorded values. The two ranked examples reproduce all
100 context factors and E*S*C utility products.

These findings support engineering reproducibility of stored components.
Hash-derived context values are not measured biological context fit. Synthetic
gene survival is not empirical sample resampling. Final audit statuses need
their own interpretation; their mere presence does not establish biological
truth. Historical risk panels do not supply the new paired unaudited-text
comparison. Human-label authenticity and agreement are separate requirements.

Five R1 working-figure metadata records are inventoried separately. In
particular, the objective P2B Fig. 2 is a different experiment from submitted
legacy Fig. 2. The legacy issues do not erase P2B's negative result.

## External metadata correction

The bundle includes metadata prefixes for 29 GEO records: three series and
26 samples. No expression rows or new pathway outcomes were downloaded or
analyzed. The supplied prefixes and manifest are hashed. Runtime is offline.

| Study | Checked public design | Permitted interpretation |
| --- | --- | --- |
| GSE52778 | Four donor-derived cell lines. All four treatments are represented for each donor. Dex versus control vehicle uses eight paired samples at 18 h. Albuterol and combined treatment are excluded from this contrast. | A new non-TP53 application with four independent donor units. |
| GSE34313 | Four public controls, three 24 h dex samples and three 4 h dex samples. The original array experiment used HASM1. Two original samples were excluded by the depositing study. | A limited cross-study culture-level replication case. Replicate suffixes do not prove pairing or independent donors. |
| GSE96583 | Series metadata alone was inspected. | Remains an unqualified fallback until donor/cell annotation and count inputs are checked. It is not automatically substituted. |

GSE34313's original description of four replicates per group must not be used
as the actual public analysis census. The earlier proposed 24 h validation has
three treated and four control samples. Keep the depositing study's exclusions
and document them. Do not retrieve the omitted samples or add outcome-based
exclusions to improve results.

The two dex studies have different accessions, platforms and publications.
Public metadata does not establish donor disjointness between studies. Any
external replication claim must carry this limitation and cannot become a
multi-donor generalization claim from the HASM1 cultures.

Primary sources:

- [GSE52778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52778)
- [GSE34313](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE34313)
- [Masuno et al., PMID 21257922](https://pmc.ncbi.nlm.nih.gov/articles/PMC3175579/): array experiment uses HASM1; ethanol vehicle.
- [Himes et al., PMID 24926665](https://pmc.ncbi.nlm.nih.gov/articles/PMC4057123/): four donor-derived lines; 18 h dex/control-vehicle design.

The original papers' abstracts/results, including KLF15 and CRISPLD2 and
published CRISPLD2 cross-study results, were visible during methods review.
R02 is not preregistered or wholly outcome-naive. No new Hallmark replication
outcomes have been computed or loaded. Do not select gene endpoints from those
visible results and label them independent held-out discoveries.

## Next analysis decision

Proceed with bounded semantic development and a versioned adapter for explicit
study contrasts. V16.1's TCGA/TP53-specific contract must not be populated with
dex samples by renaming its fields. Preserve the existing deterministic adapter
as the simple reporting control.

Before opening new expression/pathway outcomes, specify probe-to-gene mapping,
normalization, duplicate probes, the measured gene universe, the culture-level
model, the candidate census, and replication endpoints. Freeze the comparison
after development, including the exact natural LLM text and strong simple
controls. Published biological plausibility does not substitute for that test.

No new expert packet or 620-row grading task is required for R02. Reconsider a
small independent text review only after the implementation and protocol are
stable and its remaining scientific role is explicit.
