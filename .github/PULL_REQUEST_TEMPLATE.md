<!--
Say what this changes and why. If it fixes an issue, link it.
The boxes below are the four things that are easy to get wrong in this
repository; CONTRIBUTING.md explains each one.
-->

## Checks

- [ ] `uv run pre-commit run --all-files` passes
- [ ] `uv run pytest --run-slow` passes

## Does this change simulated results?

<!--
A new parameter default, an edited Kraus operator, or a different channel
composition order moves users' error rates even when the API is unchanged.
Answer "no" or describe what moves.
-->

## `test_paper_consistency.py`

- [ ] I did not touch it, **or** I changed it and cited the paper's table /
      equation / appendix number above

## `# type: ignore`

- [ ] I did not add one, **or** I added one and said above what it defers and why
- [ ] If I fixed something a frozen ignore was hiding, I removed that ignore
