# Releasing MUYAH-CODE

Releases are automatic. You change the version, tag it, and push. GitHub Actions then:
1. runs the tests
2. builds the package
3. publishes it to PyPI
4. creates a GitHub Release

Users get the new version with `pipx upgrade muyah-code`.

## One-time setup

1. **Create a PyPI account** at https://pypi.org/account/register/ and turn on 2FA (PyPI requires it).
2. **Register a trusted publisher** at https://pypi.org/manage/account/publishing/, under "Add a new pending publisher". This is what lets GitHub publish without a stored password or token.
   - PyPI project name: `muyah-code`
   - Owner: `MUYAHGaious`
   - Repository: `muyah-code`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. **Create the `pypi` environment on GitHub:** repo → Settings → Environments → New environment → `pypi`. You can require your own approval there for extra safety.

## Every release

1. Update `__version__` in `muyah_code/__init__.py` (e.g. `1.0.0` → `1.1.0`):
   - **PATCH** (1.0.1): bug fixes only.
   - **MINOR** (1.1.0): new features that keep working the old way.
   - **MAJOR** (2.0.0): something that breaks how people used it.
2. Add a section at the top of `CHANGELOG.md`.
3. Check everything passes: `python -m pytest -q` and `python -m ruff check muyah_code tests backend`.
4. Commit, tag and push:
   ```bash
   git commit -am "Release v1.1.0"
   git tag v1.1.0
   git push origin main --tags
   ```
5. Watch the run under the repo's **Actions** tab. When it's green, the release is live on PyPI.

The release workflow refuses to publish if the tag doesn't match `__version__`. A version number can only be published to PyPI once, so if something goes wrong, fix it and release the next PATCH version.
