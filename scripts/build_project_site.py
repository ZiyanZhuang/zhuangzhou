"""Stage only the public project page and its assets; standard library only."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def build_site(output: Path):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('The site output directory must be empty')
    module = ast.parse((ROOT / '__init__.py').read_text(encoding='utf-8-sig'))
    version = next(ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == '__version__'
                           for target in node.targets))
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    updated = subprocess.check_output(['git', 'log', '-1', '--format=%cI'], cwd=ROOT, text=True).strip()
    files = {
        'index.html': 'docs/index.html',
        'site.css': 'docs/site.css',
        'site.js': 'docs/site.js',
        'assets/zhuangzhou-terminal.png': 'docs/assets/zhuangzhou-terminal.png',
        'assets/zhuangzhou.png': 'zhuangzhou.png',
    }
    for target, source in files.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, destination)
    (output / 'build.json').write_text(json.dumps({
        'version': version, 'revision': revision, 'updated': updated,
    }, indent=2) + '\n', encoding='utf-8')
    (output / '.nojekyll').touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build_site(parser.parse_args().output)
    print('Public project site staged successfully')
