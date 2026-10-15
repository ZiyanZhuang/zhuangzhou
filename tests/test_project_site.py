"""Check the artifact that Pages publishes, without browsers or network access."""
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_project_site', ROOT / 'scripts/build_project_site.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.references = []
        self.has_viewport = False
        self.image_alts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for attribute in ('href', 'src'):
            if attrs.get(attribute):
                self.references.append(attrs[attribute])
        if tag == 'meta' and attrs.get('name') == 'viewport':
            self.has_viewport = True
        if tag == 'img':
            self.image_alts.append(attrs.get('alt', ''))


def test_published_site_is_standalone_and_preserves_original_illustration(tmp_path):
    assert (ROOT / 'docs/index.html').read_text(encoding='utf-8').lower().startswith('<!doctype html>')
    output = tmp_path / 'site'
    builder.build_site(output)
    html = (output / 'index.html').read_text(encoding='utf-8')
    assert html.lower().startswith('<!doctype html>')
    page = PageLinks()
    page.feed(html)
    assert page.has_viewport
    assert len(page.ids) == len(set(page.ids))
    assert page.image_alts and all(page.image_alts)
    for reference in page.references:
        url = urlsplit(reference)
        repository = 'https://github.com/ZiyanZhuang/zhuangzhou/'
        if reference.startswith(repository + 'tree/main/') or reference.startswith(repository + 'blob/main/'):
            source = reference.split('/main/', 1)[1]
            assert (ROOT / unquote(source)).exists(), reference
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert not url.path.startswith('/'), 'Resources must work under /zhuangzhou/'
            assert (output / unquote(url.path)).is_file(), reference
        elif url.fragment:
            assert unquote(url.fragment) in page.ids, reference
    assert (output / 'assets/zhuangzhou.png').read_bytes() == (ROOT / 'zhuangzhou.png').read_bytes()
    assert {p.name for p in output.iterdir()} == {
        'index.html', 'site.css', 'site.js', 'assets', 'build.json', '.nojekyll',
    }
    metadata = json.loads((output / 'build.json').read_text(encoding='utf-8'))
    from zhuangzhou import __version__
    assert metadata['version'] == __version__
    assert len(metadata['revision']) == 40
    assert metadata['updated']


def test_site_builder_refuses_to_publish_existing_unrelated_files(tmp_path):
    (tmp_path / 'private-note.txt').write_text('test fixture', encoding='utf-8')
    with pytest.raises(ValueError, match='must be empty'):
        builder.build_site(tmp_path)
    assert (tmp_path / 'private-note.txt').read_text(encoding='utf-8') == 'test fixture'
