# Contributing to Ariadne

Ariadne follows strict commit conventions to keep history readable, reviewable,
and issue-scoped.

---

# Git Commit Guidelines

Each commit message consists of:

- Header (required)
- Body (optional but strongly encouraged)
- Footer (optional)

No line may exceed 100 characters.

## Commit Message Format

| type(scope): subject |
| ---- |
| Body |
| Footer |

The header is mandatory. The scope is optional but strongly recommended.

## Revert

If a commit reverts a previous commit, use the following format:

revert: <original header>

In the body include:

This reverts commit <hash>.

## Type

Must be one of the following:

feat:     A new feature
fix:      A bug fix
refactor: A code change that neither fixes a bug nor adds a feature
perf:     A code change that improves performance
test:     Adding or updating tests
docs:     Documentation-only changes
style:    Formatting, whitespace, or CSS-only changes (no logic changes)
chore:    Build process or tooling changes
content:  A change in user-visible content
ci:       CI/CD workflow changes

## Scope

Scope should describe the part of the system affected.

Approved scopes for Ariadne:

api
indexer
embedder
db
web
ops
docs
ci
repo

Use lowercase.

## Subject

The subject must:

- Use imperative present tense ("add", not "added" or "adds")
- Not capitalize the first letter
- Not end with a period
- Be concise and descriptive

## Body

Use imperative present tense.

Explain:

- What changed
- Why it changed
- How it differs from previous behavior

Keep lines under 100 characters.

## Footer

Used for:

- Referencing GitHub issues
- Declaring breaking changes

Issue reference examples:

Closes #12
Refs #14

Breaking changes must begin with:

BREAKING CHANGE:

# Branching Strategy

- One branch per issue
- Branch name format: `feature/<number>-short-description`
- Merge via Pull Request
- Squash only if commits are noisy
- Do not push directly to `main`

# Issue Workflow

Every change starts from exactly one GitHub issue.

The issue must stay narrowly scoped and include:

- Summary
- Scope
- Acceptance Criteria
- Notes (optional)

Do not bundle unrelated work under one issue. If the work splits into multiple
independent changes, create separate issues and branches.

## Delivery Flow

Use the following workflow for every contribution:

1. Create or choose one GitHub issue.
2. Confirm the issue stays narrow enough to review in one pull request.
3. Update local `main` from `origin/main` so the issue branch starts from the
   current mainline state.
4. Branch from `main` using `feature/<number>-short-description`.
5. Implement only the work required for that issue.
6. Run the smallest relevant local validation before pushing.
7. Push the branch and open a pull request against `main`.
8. Reference the issue in the pull request and commit footer using
   `Closes #<number>` when the change completes the issue.
9. After the pull request merges, confirm the issue is closed and delete the
   completed branch locally and on GitHub.

## Pull Request Expectations

Every pull request should map to a single GitHub issue and should remain small,
reviewable, and honest about what is implemented now versus what is only
roadmap work.

Pull requests should:

- summarize what changed
- explain why it changed
- list the validation used
- call out follow-up work separately instead of bundling it
- avoid overstating readiness, feature completeness, or future roadmap items
