# Project website

Public URL: <https://ziyanzhuang.github.io/zhuangzhou/>.
English entry: <https://ziyanzhuang.github.io/zhuangzhou/?lang=en>.

The GitHub `blob/main/docs/index.html` URL displays source code; it is not the
published website. Before this change the public Pages URL returned HTTP 404 and
the repository Pages API reported no configured site.

## Source and publishing

- `docs/index.html`: semantic, bilingual project page. Original illustration,
  real terminal rendering, module links, installation, engineering evidence and
  project/supervisor credits. No new Logo or scientific-result claims.
- `docs/site.css`: responsive layouts and system fonts. No remote font service,
  canvas animation or frontend framework.
- `docs/site.js`: language selection (`?lang=en`), clipboard feedback and build
  provenance. Core Chinese content, links and native installation disclosures
  remain usable without JavaScript.
- `scripts/build_project_site.py`: standard-library staging into an empty output
  directory. Only the three page files, the two selected images, `.nojekyll`
  and `build.json` are published. The original illustration is copied unchanged;
  no duplicate is added to tracked assets. Existing output directories containing
  files are rejected, so unrelated local files cannot enter a deployment.
- `.github/workflows/pages.yml`: official GitHub Pages artifact deployment on
  `main`, with a ten-minute job limit and bounded deployment concurrency. No
  `gh-pages` branch, model API or scientific runtime is involved.

`build.json` contains only package version, Git revision and the source commit's
committer timestamp. The displayed update time is UTC, not a release date or
experiment date. A clean CI checkout identifies the exact source revision.
Local uncommitted edits are not reflected in the Git timestamp. Missing or invalid
metadata leaves a working commit-history link and a visible diagnostic message.

The repository's **Settings → Pages → Source** must select **GitHub Actions**.
Deployment uses the `github-pages` environment. See GitHub's
[custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Local validation

From the repository root, use an empty output directory:

```text
python -m pytest -q tests/test_project_site.py
python scripts/build_project_site.py --output .engineering-artifacts/project-site
python -m http.server 8765 --bind 127.0.0.1 --directory .engineering-artifacts/project-site
```

Open `http://127.0.0.1:8765/`. Serve the staged directory rather than the entire
checkout. Rebuild into a new empty directory after source changes. Tests validate
the actual staged artifact, local assets, anchors, source-code links, metadata and
the unchanged original image; browser checks cover interactive and visual behavior.
