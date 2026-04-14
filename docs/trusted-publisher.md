# Trusted Publisher Setup

This repository is configured to publish to PyPI from the `release` branch through GitHub Actions and PyPI Trusted Publishers.

## PyPI Pending Publisher values

Open the PyPI page for adding a pending publisher and use these exact values:

- `PyPI Project Name`: `mleml`
- `Owner`: `art22017`
- `Repository name`: `MLeml`
- `Workflow name`: `publish.yml`
- `Environment name`: `pypi`

The workflow filename refers to:

```text
.github/workflows/publish.yml
```

## GitHub repository settings

Create a GitHub Actions environment named `pypi` in the `MLeml` repository settings.

Recommended:

- keep the environment dedicated to publishing only
- add reviewers if you want a manual approval gate before a release is published

## Branch-based publishing model

Publishing is intentionally restricted to pushes on the `release` branch.

- pushes to `main` run normal CI only
- pushes to `release` run tests, build the package, validate metadata, and publish to PyPI via OIDC

## Important operational note

PyPI does not allow re-uploading the same version. Before pushing to `release`, bump the package version in `pyproject.toml`.

Typical flow:

1. Update `version` in `pyproject.toml`
2. Commit on `main`
3. Merge or cherry-pick that commit into `release`
4. Push `release`
5. GitHub Actions publishes the new version to PyPI

