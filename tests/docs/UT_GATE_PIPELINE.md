# UT Gate Pipeline
**Updated on September 20, 2026**

## What it is

A gate job that runs the whole unit test suite on your MR and blocks the merge if it does
not pass. It runs `tests/ut_CI_check/run_omniinfer_ut_pipeline.sh` on a CI machine, against
the code baked into the CI image — you do not run it yourself.

It checks three things, and **all three must pass**:

| Check | Requirement |
|---|---|
| Total coverage | >= 80% |
| Incremental coverage (changed lines only) | >= 80% |
| Newly failing cases | 0 |

Tests run in 4 containers in parallel, with a 20 minute timeout.

## How to use it

**Before pushing**, run the same three checks locally with the developer script:

```bash
bash tests/ut_CI_check/run_omniinfer_ut_developer.sh \
  /path/to/your/omniinfer <image_name> <base_commit>
```

| Argument | Meaning |
|---|---|
| `local_omniinfer_path` | your omniinfer checkout; must be a git repo, uncommitted changes included |
| `image_name` | full docker image name, the same one `tests/run_docker.sh` takes |
| `base_commit` | baseline for incremental coverage, e.g. `HEAD~1` or `upstream/develop` |
| `work_dir` | optional; where reports are written, default `/tmp/omniinfer_ut_selftest/ut` |

Optional environment variables: `TIMEOUT_MIN` (default 20), `MIN_COVER` (80), `MIN_DIFF`
(80), `STAGE_BASE` (staging dir, must be under `/tmp`, `/data/CI` or `/data/models`),
`PIP_INDEX`. Run the script with no arguments to print this usage.

It prints a single PASS/FAIL summary and the paths to the reports. If it passes, the gate
will too.

**If the gate fails**, read these artifacts from the job (they are archived under `ut/`):

| File | Tells you |
|---|---|
| `run_fail.log` | which cases newly failed (`new_failures: <n>`) |
| `coverage_from_dockers/coverage_report.txt` | total coverage, per file |
| `coverage_from_dockers/diffcov.html` | which of your changed lines are uncovered |
| `install_logs/DT_*.log` | full pytest output of each container |

Then reproduce locally with the developer script and fix.
