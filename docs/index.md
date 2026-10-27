# LLM-PathwayCurator

<div class="lpc-brand">
  <img src="assets/LLM-PathwayCurator_logo.png" width="90" height="90" alt="LLM-PathwayCurator logo">
  <p>Create source-linked pathway reports and inspect draft wording beside the
  enrichment evidence. The current default runs without a language model.</p>
</div>

[![Docs](https://img.shields.io/badge/docs-latest-blue?style=flat&logo=readthedocs&logoColor=white)](https://llm-pathwaycurator.readthedocs.io/)
[![Historical preprint](https://img.shields.io/badge/bioRxiv-historical_preprint-B31B1B?style=flat)](https://doi.org/10.64898/2026.02.18.706381)
[![License: MIT](https://img.shields.io/badge/License-MIT-f1c40f?style=flat)](https://github.com/kenflab/LLM-PathwayCurator/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-555555?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)

---

Start with [Getting started](getting-started.md). The [User guide](user-guide.md)
specifies the input tables and decisions. [Concepts](concepts.md) distinguishes
statistical reporting, draft inspection and independent biological validation.

The [structured curation guide](structured-curation.md) adds optional
supporting-gene modules, typed observations and saved model- or human-authored
proposals to the same source workflow. Module membership does not change
statistical eligibility or certify the meaning of a proposal.

The current repository source includes this workflow. Previously published
packages and the [historical preprint](https://doi.org/10.64898/2026.02.18.706381)
describe an earlier pipeline. Exact historical paper reproduction uses a pinned
commit, not an unqualified checkout of current main.

The development and manuscript analyses have their own
[revision entry point](https://github.com/kenflab/LLM-PathwayCurator/tree/main/paper/revision/CRM_R1).
