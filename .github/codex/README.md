# ApxInf automated code review

This directory contains the trusted policy, schema, and deterministic scripts
used by `.github/workflows/apxinf-ai-review.yml`.

## Trust model

- The workflow runs through `pull_request_target` and checks out only the exact
  base SHA. Its prompt, scripts, and installed skill therefore come from the
  trusted base branch.
- The PR head is fetched as Git objects for `git diff` and `git show`. It is
  never checked out, built, imported, sourced, or executed.
- The `analyze` job has read-only GitHub permissions and invokes Codex once
  with the `:read-only` permission profile.
- The separate `publish` job receives only validated artifacts and is the only
  job with permission to update a PR comment. It refuses to publish if the PR
  head changed after analysis.
- Same-repository PRs run automatically. Fork PRs require a maintainer to add
  the `ai-review` label.

## Repository setup

1. Add an Actions secret named `OPENAI_API_KEY` under **Settings → Secrets and
   variables → Actions**.
2. Merge the workflow into the default branch. GitHub does not run a newly
   introduced `pull_request_target` workflow from the PR that introduces it.
3. Create the `ai-review` label for explicitly authorized fork reviews.
4. Open a small same-repository test PR. Confirm that the `analyze` job uploads
   one JSON/Markdown artifact and the `publish` job creates or updates one
   comment containing `<!-- apxinf-ai-review -->`.
5. Keep the workflow advisory while evaluating false positives. Do not make it
   a required check or automatically request changes until enough real PRs have
   been compared with maintainer reviews.

## Cost controls

- One Codex invocation per PR head.
- Concurrency cancellation discards superseded runs after a new push.
- High reasoning effort is selected only for runtime, model, native, Rust,
  CUDA, HIP, or Metal paths; other changes use medium effort.
- The prompt contains only task constraints. Review rules live in the installed
  repository-pinned skill, and Markdown is rendered deterministically from one
  structured JSON response.

## Updating the skill

Update `.github/skills/apxinf-code-review/` in a separately reviewed PR. The
workflow always installs the skill from the base revision, so a PR cannot alter
the policy used to review itself.
