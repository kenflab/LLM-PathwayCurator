# API reference

## Public reporting API

::: llm_pathway_curator.review.ReviewConfig

::: llm_pathway_curator.review.ReviewResult

::: llm_pathway_curator.review.review_enrichment

## Pipeline compatibility

::: llm_pathway_curator.pipeline.RunConfig

::: llm_pathway_curator.pipeline.run_pipeline

The default `workflow="source"` delegates to the public reporting API.
`workflow="legacy"` explicitly requests the historical pipeline. Study-specific
revision prototypes are not installed or exposed as a public API.
