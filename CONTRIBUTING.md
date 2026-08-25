# Contributing

Thanks for looking at DualYbSim. This page covers the setup, and the three rules
the code cannot tell you on its own.

The library is pre-1.0 and the paper it implements is still in preparation, so
both the public API and the tabulated parameter values may change. Please open an
issue before starting anything large, so we can agree on the shape first.

## Setup

[uv](https://docs.astral.sh/uv/) is the only supported development path.

```bash
uv sync                    # .venv from uv.lock: the library, editable, plus the dev tooling
uv run pre-commit install  # ruff (lint, format and docstrings), mypy and bandit on commit
uv run pytest --run-slow   # the suite CI runs
```

`uv sync` does not need a Python interpreter on the machine: it reads the `3.13`
pin in `.python-version` and fetches that toolchain. Nothing needs activating —
every `uv run` uses that environment.

Two things to know about the checks:

- **CI runs `pre-commit run --all-files` as its own job**, so a hook you skipped
  locally fails the build. If you did not install the hooks, run
  `uv run pre-commit run --all-files` before pushing.
- **`--run-slow` is not optional in CI.** It adds the Kraus-level twirling
  checks; plain `uv run pytest` skips them.

After changing a dependency, `uv lock` regenerates `uv.lock`. Commit it: CI
installs with `uv sync --locked` and fails if the two have drifted.

## Three rules you cannot guess from the code

### 1. Never make `test_paper_consistency.py` pass by editing the expected value

Every number in `tests/test_paper_consistency.py` is transcribed from the paper's
noise-model appendix rather than read back from the code. A failure therefore
means one of exactly two things:

- the code drifted from the paper — fix the code; or
- the transcription was wrong — fix the test, and cite the paper's table,
  equation or appendix number in the pull request.

Editing an expected value to match new behaviour, with no citation, silently
unpins the library from the paper it implements. It is the one change a reviewer
cannot catch by reading the diff, which is why it needs the citation.

Divergences from the paper that are on purpose live in `TestDeliberateDepartures`
(`tests/test_paper_consistency.py`). Adding one is fine; adding one without
saying why in the pull request is not.

### 2. Close every moment with a `TICK`, the last one included

Idling noise is attributed per moment, and the `TICK` is what marks a moment as
over. A circuit whose final moment has no `TICK` loses that moment's idling noise
— optimistically — and a `readout_protocol` cannot tell which qubits sat out the
measurement. The library warns when it has to drop idling for that reason; do not
silence that warning in a test or an example, fix the circuit.

### 3. Remove a `# type: ignore` when you fix what it hides

`uv run mypy src/dualybsim` is clean, but seventeen sites carry a per-site
`# type: ignore[code]` that marks a real latent bug still needing a decision:

| file | sites |
| --- | --- |
| `src/dualybsim/circuit.py` | 9 |
| `src/dualybsim/params.py` | 4 |
| `src/dualybsim/kraus/channels.py` | 2 |
| `src/dualybsim/twirling/gpta.py` | 2 |

`warn_unused_ignores` is on, so once the underlying issue is fixed mypy reports
the ignore as unnecessary — removing it is part of the same change. Adding a new
one is allowed, but say in the pull request what it defers and why.

## When your change moves the numbers

A change to a parameter default, to a Kraus operator, or to the order channels
compose in changes users' simulated error rates even when the API is untouched.
Anyone reproducing published numbers needs to see that, so call it out
explicitly in the pull request.

`examples/noise_channel_tour.ipynb` is committed with its outputs and is the
documentation for which channels attach to which operation. If you change what
the circuit builder emits, re-run it or those outputs go stale:

```bash
uv sync --group notebook
uv run jupyter execute examples/noise_channel_tour.ipynb --inplace
```

`docs/channel_reference.md` and `docs/channel_ordering.md` state the same rules
in prose — check them too.

## Housekeeping

The version string lives in two places, `pyproject.toml` and
`src/dualybsim/__init__.py`. Keep them in step.

For commit messages: a short imperative subject, then prose saying why rather
than what. English, so the history reads as one document.

## Licence

Contributions are accepted under the MIT licence, the same terms as the rest of
the repository. See [LICENSE](LICENSE).
