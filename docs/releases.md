# Releases and archives

## Code, source data and upstream inputs

A GitHub release identifies a fixed repository commit. The linked Zenodo record
archives that release. Use its version-specific DOI when citing an analysis.
The manuscript's recorded analysis commits remain the provenance for upstream
fits; a later reporting release does not replace them.

| Content | Included in the repository archive | Reproduction scope |
| --- | --- | --- |
| `src/`, examples and documentation | Yes | Public tool and synthetic demonstrations |
| `paper/revision/CRM_R1/data/` and `data_manifest.json` | Yes; 64 checksummed files | Saved enrichment records, original model text, anonymous ratings and comparison results |
| Manuscript renderers and `reproduce.py` | Yes | Figures 2–3/S1–S4, numerical tables and reported summaries from saved records |
| `paper/source_data/CRM_R1_TCGA_inputs.zip` | Yes; 11 input files and their manifest | Recorded TCGA snapshot; the large expression matrix is a checksum-verified download |
| Historical split-analysis code | Yes | Verified discovery/membership/validation source code and execution identities |
| External expression matrices, original study locks and model environment | See the study-specific guides | Required when repeating upstream fits |

The [figure reproduction guide](https://github.com/kenflab/LLM-PathwayCurator/blob/main/paper/revision/CRM_R1/FIGURE_REPRODUCTION.md)
maps figures and tables to their code and input files. Figure 1 is author-supplied
artwork. A software wheel or sdist is not the manuscript archive: `MANIFEST.in`
excludes `paper/` from the Python distribution.

## Existing v0.1.0 records

| DOI | Record |
| --- | --- |
| [10.5281/zenodo.18675487](https://doi.org/10.5281/zenodo.18675487) | GitHub-generated v0.1.0 release archive |
| [10.5281/zenodo.18625777](https://doi.org/10.5281/zenodo.18625777) | Separate code, benchmarks and manuscript source-data archive |

These are separate records, not a version DOI and its all-versions DOI.
A new GitHub release is handled by the enabled GitHub integration. A new version
of the separate manuscript archive must be deposited through its own record.
The contents listed above are tracked in the repository and can accompany the
code in the GitHub-linked archive. Confirm the actual Zenodo file inventory and
metadata before choosing the DOI cited in the manuscript.

## Publish a release

1. Merge the reviewed changes and check CI on the exact main commit.
2. Run the examples and the manuscript reproduction command. Check the archived
   split-code hashes with `archive/split_replication/verify_code.py` under the
   manuscript directory.
3. Select an unused release tag, such as `v0.2.0`, consistent with `CITATION.cff`.
   Python package versions come from Git tags through `setuptools-scm`.
4. Confirm that the repository is enabled on the Zenodo GitHub integration page.
5. Tag the reviewed commit and publish a GitHub Release for that tag. A pushed
   tag alone is not the Zenodo release-archiving step.
6. Wait for Zenodo to finish. Verify the version, authors, license, description
   and archived source files, and record the new version DOI.
7. Update the manuscript's code availability statement and repository citation
   links with that verified DOI. Keep the original analysis commits in Methods.

This repository also starts PyPI and GHCR publishing workflows on version-tag
pushes. PyPI publishes stable tags; the container workflow builds both supported
architectures. Check those workflows along with the GitHub release and Zenodo
record. The general Docker environment is not the recorded R environment for
all manuscript fits.

`CITATION.cff` provides the release metadata; `.zenodo.json` is unnecessary for
this setup. Do not put a previous version DOI into a new release as if it were
the new archive's identifier. When needed, provide an additional data deposit
for upstream files that are not in the repository, preserving their recorded
identities and linking that deposit from the software record.

## Source archive versioning

`.git_archival.txt` and its `export-subst` attribute carry version information
into a Git source archive. Packaging CI compares wheels built from the checkout
and from `git archive`, to check that the tested source archive retains the checkout version. The container build takes its package version from the release tag.

Official documentation:

- [Enable a Zenodo repository](https://help.zenodo.org/docs/github/enable-repository/)
- [Archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/)
- [Zenodo versioning](https://help.zenodo.org/docs/deposit/manage-versions/)
- [CITATION.cff metadata](https://help.zenodo.org/docs/github/describe-software/citation-file/)
