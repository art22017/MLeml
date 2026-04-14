# Release and Publishing Guide

## 1. Prepare the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

If you are using GitHub Actions Trusted Publishers, configure PyPI first as described in [trusted-publisher.md](trusted-publisher.md).

## 2. Run validation

Fast tests:

```bash
pytest -s -m "not slow"
```

Example tests:

```bash
pytest -s -m slow
```

Build validation:

```bash
python -m build
python -m twine check dist/*
```

## 3. Commit and tag

```bash
git add .
git commit -m "Initial MLeml release"
git tag v0.1.0
```

## 4. Release branch publishing

The repository is configured to publish to PyPI from the `release` branch.

Typical branch flow:

```bash
git checkout main
git pull
# bump version in pyproject.toml
git commit -am "Release v0.1.0"
git checkout -B release
git push -u origin release
```

Every publishable push to `release` must carry a new version number.

## 5. Push to GitHub manually

If the GitHub repository already exists:

```bash
git remote add origin git@github.com:<your-user>/MLeml.git
git push -u origin main
git push origin v0.1.0
```

HTTPS variant:

```bash
git remote add origin https://github.com/<your-user>/MLeml.git
git push -u origin main
git push origin v0.1.0
```

## 6. Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

Install test:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mleml
```

## 7. Upload to PyPI

```bash
python -m twine upload dist/*
```

## 8. Post-release checks

- verify the project page renders correctly
- verify `pip install mleml`
- create a GitHub release from the tag
- copy the README quick-start snippets into the release notes if useful
