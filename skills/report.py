"""
Report Generation Skills
=========================
Registers skills for generating Word (.docx) reports from pipeline results.
"""

import json
import logging
import os

from .registry import Skill, SkillParam, SkillRegistry

logger = logging.getLogger(__name__)


def _normalize_esm2_results(d: dict, source_dir: str = None) -> dict:
    """
    Normalize ESM-2 scan JSON output to the format expected by build_mutation_report.

    The esm_mutation_scan.py script outputs keys like 'protein', 'total_beneficial',
    'delta_ll' in mutations — but report templates expect 'protein_name',
    'top_mutations', 'n_positions_scanned', 'score'/'recommendation' in mutations.

    This bridges the gap without modifying the scan script.
    """
    d = dict(d)  # shallow copy

    # ── 1. protein name ─────────────────────────────────────────────────────────
    import re as _re
    if not d.get('protein_name') and d.get('protein'):
        raw = str(d['protein'])
        # Strip chain suffixes: "6J_chain_A" → "6J", "CYP6AB328_chainA" → "CYP6AB328"
        clean = _re.sub(r'[_\s]chain[_\s]?[A-Za-z]$', '', raw, flags=_re.IGNORECASE).strip()
        candidate = clean or raw
        # If the result looks like a bare PDB ID (≤4 alphanumeric chars),
        # prefer the source_dir stem (e.g. "CYP6AB328_esm2_scan" → "CYP6AB328")
        if source_dir and _re.match(r'^[0-9A-Za-z]{1,4}$', candidate):
            dir_stem = os.path.basename(os.path.abspath(source_dir))
            better = _re.sub(r'[_\-]esm2[_\-]?scan$', '', dir_stem, flags=_re.IGNORECASE)
            better = _re.sub(r'[_\-]scan$', '', better, flags=_re.IGNORECASE)
            better = better.strip()
            if better and better != dir_stem:
                candidate = better
        d['protein_name'] = candidate

    # ── 2. sequence length → n_positions_scanned ────────────────────────────────
    if not d.get('n_positions_scanned'):
        seq = d.get('sequence', '')
        if seq:
            d['n_positions_scanned'] = len(seq)

    # ── 3. Normalize mutations: add 'score' alias and 'recommendation' ──────────
    mutations = d.get('mutations', [])
    if mutations:
        normalized = []
        for m in mutations:
            m = dict(m)
            # 'score' is what the non-enriched table column reads
            if 'score' not in m:
                m['score'] = m.get('delta_ll', 0)
            # 'recommendation' drives the "promising" count in abstract
            if 'recommendation' not in m:
                dl = m.get('delta_ll', 0)
                m['recommendation'] = 'Promising' if dl > 0.5 else 'Neutral'
            normalized.append(m)
        d['mutations'] = normalized
        mutations = normalized

    # ── 4. top_mutations ────────────────────────────────────────────────────────
    if not d.get('top_mutations') and mutations:
        d['top_mutations'] = sorted(
            mutations, key=lambda x: -x.get('delta_ll', x.get('score', 0))
        )[:10]

    # ── 5. Generate figures if missing ──────────────────────────────────────────
    if not d.get('figures') and source_dir and mutations:
        figs = _generate_esm2_figures(mutations, d.get('sequence', ''), source_dir,
                                      d.get('protein_name', 'Protein'))
        if figs:
            d['figures'] = figs

    # ── 6. Auto-fetch literature if missing ─────────────────────────────────────
    if not d.get('literature') and d.get('protein_name'):
        papers = _auto_fetch_literature(d['protein_name'], d.get('organism', ''))
        if papers:
            d['literature'] = papers
            logger.info("Auto-fetched %d papers for %s", len(papers), d['protein_name'])

    # ── 7. Auto structural enrichment if mutations lack ss_type ─────────────────
    if mutations and not mutations[0].get('ss_type'):
        _seq = d.get('sequence', '')
        _pname = d.get('protein_name', '')
        # Infer active sites via P450 motif search + CDD
        _site_info = _infer_cyp_active_sites(_seq, _pname) if _seq else {}
        _active_site = _site_info.get('active_site', set())
        _catalytic   = _site_info.get('catalytic', set())
        # Write active site annotations into results for report use
        if _site_info.get('annotations'):
            d['active_site_annotations'] = {
                str(k): v for k, v in _site_info['annotations'].items()
            }
        enriched = _auto_enrich_structure(
            mutations=mutations,
            protein_name=_pname,
            sequence=_seq,
            pdb_path=d.get('pdb_path'),
            source_dir=source_dir,
            active_site=_active_site,
            catalytic=_catalytic,
        )
        if enriched:
            d['mutations'] = enriched
            # Rebuild top_mutations from enriched data
            d['top_mutations'] = sorted(
                enriched, key=lambda x: -x.get('delta_ll', x.get('score', 0))
            )[:10]
            logger.info("Auto structural enrichment applied (%d mutations)", len(enriched))

    # ── 8. Auto fusion protein detection ──────────────────────────────────────
    _seq = d.get('sequence', '')
    _pname = d.get('protein_name', '')
    if _seq and not d.get('fusion_info'):
        try:
            from zhuangzhou.mutation_enricher.domain_annotator import detect_fusion_domains
            fusion_info = detect_fusion_domains(_seq, _pname)
            if fusion_info.get('is_fusion'):
                d['fusion_info'] = fusion_info
                logger.info("Fusion protein detected: %d domains, %d linkers",
                            len(fusion_info.get('domain_boundaries', [])),
                            len(fusion_info.get('linkers', [])))
        except Exception as e:
            logger.debug("Fusion detection skipped: %s", e)

    return d


def _generate_esm2_figures(mutations: list, sequence: str, output_dir: str,
                            protein_name: str) -> dict:
    """
    Generate ESM-2 visualization figures (bar chart of top mutations).
    Returns dict with figure paths, or empty dict if matplotlib unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return {}

    figs = {}

    # ── Bar chart: top 20 mutations by delta-LL ─────────────────────────────────
    try:
        top = sorted(mutations, key=lambda x: -x.get('delta_ll', x.get('score', 0)))[:20]
        labels = [m.get('mutation', '?') for m in top]
        scores = [m.get('delta_ll', m.get('score', 0)) for m in top]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2166AC' if s > 1.0 else '#74ADD1' if s > 0.5 else '#ABD9E9'
                  for s in scores]
        bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xlabel('Delta Log-Likelihood (ΔLL)', fontsize=11)
        ax.set_title(f'{protein_name} — Top ESM-2 Beneficial Mutations', fontsize=12)
        ax.tick_params(axis='y', labelsize=9)
        plt.tight_layout()
        bar_path = os.path.join(output_dir, 'esm2_top_mutations.png')
        fig.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figs['top_mutations'] = bar_path
        logger.info("Generated top_mutations figure: %s", bar_path)
    except Exception as e:
        logger.warning("Failed to generate top_mutations figure: %s", e)

    # ── Heatmap: per-position max delta-LL ──────────────────────────────────────
    try:
        if sequence and len(sequence) <= 800:
            pos_max = {}
            for m in mutations:
                pos = m.get('position', 0)
                dl = m.get('delta_ll', 0)
                pos_max[pos] = max(pos_max.get(pos, 0), dl)

            x = list(range(1, len(sequence) + 1))
            y = [pos_max.get(i, 0) for i in x]

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.bar(x, y, width=1.0, color='#4393C3', alpha=0.8)
            ax.set_xlabel('Sequence Position', fontsize=10)
            ax.set_ylabel('Max ΔLL', fontsize=10)
            ax.set_title(f'{protein_name} — ESM-2 Mutational Landscape', fontsize=11)
            ax.set_xlim(0, len(sequence) + 1)
            plt.tight_layout()
            landscape_path = os.path.join(output_dir, 'esm2_landscape.png')
            fig.savefig(landscape_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            figs['landscape'] = landscape_path
            logger.info("Generated landscape figure: %s", landscape_path)
    except Exception as e:
        logger.warning("Failed to generate landscape figure: %s", e)

    return figs


# ── Structural enrichment helpers ────────────────────────────────────────────

def _auto_fetch_literature(protein_name: str, organism: str = '') -> list:
    """Fetch PubMed papers for a protein with multi-level family fallback.

    Strategy (cascade — stops when ≥3 papers found):
      1. Exact protein name
      2. Gene subfamily (CYP6AB328 → CYP6AB)
      3. Protein family + organism (e.g. "cytochrome P450 reductase Chlamydomonas")
      4. Broad protein family (e.g. "cytochrome P450" or "oleic acid synthase")
      5. For fusion proteins: search each component separately

    For novel/less-known proteins, these cascaded searches ensure the report
    always has relevant literature context.
    """
    import re as _re
    try:
        from zhuangzhou.huizi import LiteratureFetcher
        lf = LiteratureFetcher()
        papers = []
        seen_pmids = set()

        def _add(query, analyses=None, max_r=5):
            nonlocal papers, seen_pmids
            more = lf.fetch(query, analyses or ['esm2_scan', 'mutation_enrichment'], max_results=max_r)
            for p in more:
                pmid = p.get('pmid')
                if pmid and pmid not in seen_pmids:
                    papers.append(p)
                    seen_pmids.add(pmid)

        # --- 1. Exact protein name ---
        _add(protein_name)

        # --- 2. For fusion proteins, search each component separately ---
        # Detect fusion from name patterns: "ProtA_GSGx2_ProtB", "ProtA|ProtB"
        fusion_parts = []
        for sep_pat in [r'_GSG\w*_', r'_GGGGS\w*_', r'_linker_', r'\|', r'\+']:
            parts = _re.split(sep_pat, protein_name, flags=_re.IGNORECASE)
            if len(parts) >= 2:
                fusion_parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
                break
        if fusion_parts and len(papers) < 5:
            for part in fusion_parts[:3]:
                if len(papers) >= 5:
                    break
                _add(part, max_r=3)

        # --- 3. Gene subfamily: strip trailing digits (CYP6AB328 → CYP6AB) ---
        if len(papers) < 3:
            family = _re.sub(r'\d+$', '', protein_name)
            if family and family != protein_name and len(family) >= 3:
                _add(family)

        # --- 4. Protein family + organism ---
        if len(papers) < 3 and organism:
            # Extract short organism (genus only)
            genus = organism.split()[0] if organism else ''
            if genus and len(genus) > 2:
                _add(f'{protein_name} {genus}', max_r=3)

        # --- 5. Broad enzyme family fallback ---
        if len(papers) < 3:
            # P450 family
            cyp_match = _re.match(r'^(CYP\d+)', protein_name, _re.I)
            if cyp_match:
                _add(cyp_match.group(1) + ' cytochrome P450', ['docking', 'structure'], 4)

            # CPR (cytochrome P450 reductase)
            if _re.search(r'CPR|cpr\d*|P450.reductase', protein_name, _re.I):
                _add('cytochrome P450 reductase directed evolution', max_r=3)

            # OAS (oleic acid synthase / desaturase family)
            if _re.search(r'OAS|oleic.acid.synth|desaturase', protein_name, _re.I):
                _add('fatty acid desaturase directed evolution', max_r=3)

            # General enzyme keywords
            for kw_pat, kw_query in [
                (r'kinase', 'kinase directed evolution mutation'),
                (r'lipase', 'lipase protein engineering'),
                (r'protease', 'protease directed evolution'),
                (r'dehydrogenase', 'dehydrogenase enzyme engineering'),
                (r'transferase', 'transferase protein engineering'),
                (r'oxidase|oxygenase', 'oxidase protein engineering'),
                (r'reductase', 'reductase enzyme engineering'),
                (r'synthase|synthetase', 'synthase enzyme engineering'),
            ]:
                if _re.search(kw_pat, protein_name, _re.I):
                    _add(kw_query, max_r=3)
                    break

        return papers[:8]
    except Exception as e:
        logger.warning("Auto literature fetch failed: %s", e)
        return []


def _infer_cyp_active_sites(sequence: str, protein_name: str = '') -> dict:
    """基于 P450 家族保守 Motif 从序列推断活性位点残基。

    依据（学术标准）：
      - 血红素结合 Cys：FxxGxxxCxG — 铁血红素第五轴向配位体，全 P450 绝对保守
      - I-helix 保守 Thr：AGxDT / AGxET（~100-180 残基在 Cys 上游）—— 质子传递通道
      - Meander ExxR 保守 Arg：在 Cys 上游 10-40 残基 —— 维持 Meander β-sheet

    Gotoh (1992) J Biol Chem 267:83–90 定义了 SRS1-6 区域；
    本函数基于 motif 定位催化核心，不做 SRS 全套比对（需要序列比对工具）。

    返回:
      {
        'active_site': set[int],    # 活性位点残基编号（1-indexed）
        'catalytic':   set[int],    # 直接催化残基（活性位点子集）
        'annotations': dict[int, str],  # 位点 → 功能说明
        'method': str,              # 推断方法描述
      }
    """
    import re as _re

    if not sequence:
        return {'active_site': set(), 'catalytic': set(), 'annotations': {}, 'method': 'no_sequence'}

    annotations: dict = {}
    catalytic: set = set()

    # ── 1. 血红素结合 Cys：[FYW]xxGxxxCxG ──────────────────────────────────────
    heme_pat = _re.compile(r'[FYW].{2}G.{3}C.G', _re.I)
    heme_cys_list = []
    for m in heme_pat.finditer(sequence):
        full = m.group()
        c_off = full.upper().index('C')
        cys_1idx = m.start() + c_off + 1
        if sequence[cys_1idx - 1].upper() == 'C':
            heme_cys_list.append(cys_1idx)
            annotations[cys_1idx] = 'Heme-binding Cys (Fe axial ligand, absolutely conserved)'
            catalytic.add(cys_1idx)

    primary_cys = heme_cys_list[0] if heme_cys_list else None

    # ── 2. I-helix 保守 Thr（质子传递通道） ────────────────────────────────────
    ihelix_pat = _re.compile(r'[AG][AGST].{1}[DE]T', _re.I)
    for m in ihelix_pat.finditer(sequence):
        thr_1idx = m.end()   # 模式末位 T 的 1-indexed
        if primary_cys:
            dist = primary_cys - thr_1idx
            if 100 <= dist <= 180 and sequence[thr_1idx - 1].upper() == 'T':
                annotations[thr_1idx] = (
                    f'I-helix conserved Thr (proton relay channel, '
                    f'{dist} residues N-terminal of heme-Cys{primary_cys})'
                )

    # ── 3. Meander ExxR 保守 Arg ────────────────────────────────────────────────
    exxr_pat = _re.compile(r'E.{2}R', _re.I)
    for m in exxr_pat.finditer(sequence):
        arg_1idx = m.start() + 4   # R 的 1-indexed
        if primary_cys:
            dist = primary_cys - arg_1idx
            if 0 < dist <= 40 and sequence[arg_1idx - 1].upper() == 'R':
                annotations[arg_1idx] = (
                    f'Meander ExxR conserved Arg '
                    f'(structural integrity of Meander β-sheet, '
                    f'{dist} residues N-terminal of heme-Cys{primary_cys})'
                )

    active_site = set(annotations.keys())

    method = 'P450_motif_sequence_search (FxxGxxxCxG + AGxDT/ET + ExxR)'
    if not active_site:
        method = 'motif_search_no_hit'
    logger.info('_infer_cyp_active_sites: %s → active_site=%s catalytic=%s',
                protein_name, sorted(active_site), sorted(catalytic))

    return {
        'active_site':  active_site,
        'catalytic':    catalytic,
        'annotations':  annotations,
        'method':       method,
    }


def _query_ncbi_cdd_sites(sequence: str, protein_name: str = '',
                           cache_dir: str = None) -> dict:
    """向 NCBI CDD 提交序列，返回功能位点注释。

    NCBI CDD (Conserved Domain Database) 收录了 pfam00067 (P450) 的
    功能位点注释，可精确映射到查询序列的残基位置。

    结果缓存于 ~/.zhuangzhou/structure_cache/<name>_cdd.json。
    返回 {'sites': {resnum: label, ...}, 'domain': str} 或空 dict（查询失败时）。
    """
    import re as _re
    import time as _time
    import urllib.request as _ur
    import urllib.parse as _up

    cache_dir = cache_dir or os.path.join(
        os.path.expanduser('~'), '.zhuangzhou', 'structure_cache'
    )
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = _re.sub(r'[^\w]', '_', protein_name or 'unknown')
    cache_file = os.path.join(cache_dir, f'{safe_name}_cdd.json')

    # 命中缓存直接返回
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    CDD_URL = 'https://www.ncbi.nlm.nih.gov/Structure/bwrpsb/bwrpsb.cgi'
    hdrs = {'User-Agent': 'Zhuangzhou/1.0', 'Content-Type': 'application/x-www-form-urlencoded'}

    # 提交
    try:
        post = _up.urlencode({
            'queries': sequence, 'db': 'cdd', 'smode': 'auto',
            'useid1': 'true', 'maxhit': '500', 'filter': 'true',
            'evalue': '0.01', 'dmode': 'full', 'tdata': 'hits',
        }).encode('ascii')
        req = _ur.Request(CDD_URL, data=post, headers=hdrs)
        with _ur.urlopen(req, timeout=15) as r:
            resp = r.read().decode()
    except Exception as e:
        logger.warning('CDD 提交失败: %s', e)
        return {}

    cdsid_m = _re.search(r'#cdsid\s+(\S+)', resp)
    if not cdsid_m:
        return {}
    cdsid = cdsid_m.group(1)

    # 轮询（最多60秒）
    result_text = ''
    for _ in range(8):
        _time.sleep(8)
        try:
            poll_params = _up.urlencode({'cdsid': cdsid, 'tdata': 'hits', 'dmode': 'full'})
            pr = _ur.Request(f'{CDD_URL}?{poll_params}', headers=hdrs)
            with _ur.urlopen(pr, timeout=15) as r:
                result_text = r.read().decode()
            if '#status 0' in result_text or '#status 3' in result_text:
                break
        except Exception:
            pass

    if '#status 0' not in result_text and '#status 3' not in result_text:
        logger.warning('CDD 查询超时或失败')
        return {}

    # 解析功能位点行（含 "Specific" 或 "Active site" 注释）
    sites = {}
    domain_name = ''
    for line in result_text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 7:
            continue
        # CDD 结果列：Query  Hit_type  PSSM_ID  From  To  E-val  Bitscore  ...
        hit_type = parts[1].strip() if len(parts) > 1 else ''
        if not domain_name and 'p450' in parts[2].lower() if len(parts) > 2 else False:
            domain_name = parts[2].strip()
        # 功能位点行通常含 "Specific" 或 "Functional" 注释
        ann = parts[-1].strip() if parts else ''
        if 'site' in ann.lower() or 'active' in ann.lower() or 'heme' in ann.lower():
            try:
                from_pos = int(parts[3])
                to_pos   = int(parts[4])
                for pos in range(from_pos, to_pos + 1):
                    sites[pos] = ann
            except (ValueError, IndexError):
                pass

    result = {'sites': sites, 'domain': domain_name}
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    logger.info('CDD 查询完成: domain=%s sites=%d', domain_name, len(sites))
    return result


def _find_huizi_pdb_path(protein_name: str = None) -> str:
    """从 HuiZi session context 读取用户已提供的本地 PDB 路径。

    遍历所有 session 文件，优先匹配蛋白质名称，其次取最新会话。
    仅返回磁盘上实际存在的路径。
    """
    huizi_dir = os.path.join(os.path.expanduser('~'), '.zhuangzhou', 'huizi')
    if not os.path.isdir(huizi_dir):
        return None

    best_path = None
    best_mtime = 0
    for session_name in os.listdir(huizi_dir):
        ctx_file = os.path.join(huizi_dir, session_name, 'session_context.json')
        if not os.path.exists(ctx_file):
            continue
        try:
            mtime = os.path.getmtime(ctx_file)
            with open(ctx_file, encoding='utf-8') as f:
                ctx = json.load(f)
            pdb_path = ctx.get('pdb_path')
            if not pdb_path or not os.path.exists(pdb_path):
                continue
            # 若蛋白名称吻合，直接返回
            ctx_name = ctx.get('protein_name') or ''
            if (protein_name and ctx_name and
                    protein_name.lower() in ctx_name.lower()):
                return pdb_path
            # 否则记录最新会话的路径
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = pdb_path
        except Exception:
            continue
    return best_path


def _auto_enrich_structure(mutations: list, protein_name: str,
                            sequence: str = None, pdb_path: str = None,
                            source_dir: str = None,
                            active_site: set = None,
                            catalytic: set = None) -> list:
    """基于已有的真实结构文件对突变列表进行结构注释。

    优先级（从高到低）：
      1. 调用方显式传入的 pdb_path（来自 results 或 protein_context）
      2. HuiZi session context 中记录的用户本地 PDB 路径
      3. AFDB 结构缓存（仅对 UniProt 可查询的蛋白适用）

    注意：**不得**对序列进行截断后折叠——截断序列的坐标无法与原始残基编号
    一一对应，用于位点注释将引入系统性错误，不符合学术规范。

    返回注释后的突变列表，或 None（结构不可用时）。
    """
    # 1. 显式传入的 pdb_path
    if not pdb_path or not os.path.exists(pdb_path):
        # 2. HuiZi session 中的用户本地结构（最高可信度）
        pdb_path = _find_huizi_pdb_path(protein_name)

    if not pdb_path or not os.path.exists(pdb_path):
        logger.info("未找到 %s 的结构文件，跳过结构富集", protein_name)
        return None

    logger.info("使用结构文件进行富集: %s", pdb_path)
    try:
        from zhuangzhou.mutation_enricher import enrich_mutations as _enrich
        enriched = _enrich(
            mutations=mutations,
            pdb_path=pdb_path,
            protein_name=protein_name,
            active_site=active_site or set(),
            catalytic=catalytic or set(),
            run_literature=False,   # 文献单独抓取
        )
        if enriched and enriched[0].get('ss_type'):
            return enriched
        logger.warning("enrich_mutations 未返回 ss_type，富集可能失败")
    except Exception as e:
        logger.warning("结构富集失败 (%s): %s", protein_name, e)
    return None


def _generate_mutation_report(results: str = None, results_json: str = None,
                              output_path: str = None, title: str = None,
                              author: str = 'Zhuangzhou',
                              protein_context: dict = None,
                              config: dict = None, **kw) -> dict:
    """
    Generate a mutation analysis Word report from pipeline results.

    Accepts either a results dict (from previous skill) or a JSON file path.
    When results_json is an esm2_beneficial_mutations.json file, the parent
    directory is automatically injected as output_dir so the report lands
    next to the scan outputs.
    """
    from zhuangzhou.report import build_mutation_report

    _json_source_dir = None

    if isinstance(results, str):
        # Could be a JSON string or file path
        if os.path.exists(results):
            _json_source_dir = os.path.dirname(os.path.abspath(results))
            with open(results, encoding='utf-8', errors='replace') as f:
                results_dict = json.load(f)
        else:
            try:
                results_dict = json.loads(results)
            except (json.JSONDecodeError, TypeError):
                return {'error': f'Cannot parse results: {results[:200]}',
                        'hint': 'Provide a valid JSON file path or JSON string'}
    elif isinstance(results, dict):
        results_dict = results
    elif results_json and os.path.exists(results_json):
        _json_source_dir = os.path.dirname(os.path.abspath(results_json))
        with open(results_json, encoding='utf-8', errors='replace') as f:
            results_dict = json.load(f)
    else:
        return {'error': 'No results provided',
                'hint': 'Pass results dict, JSON string, or results_json file path'}

    # Inject output_dir so build_mutation_report writes alongside the scan outputs
    # when no explicit output_path was requested.
    if not output_path and _json_source_dir:
        results_dict = dict(results_dict)          # don't mutate caller's dict
        results_dict.setdefault('output_dir', _json_source_dir)

    # Normalize ESM-2 scan output to report template expectations
    results_dict = _normalize_esm2_results(results_dict, source_dir=_json_source_dir)

    return build_mutation_report(results_dict, output_path=output_path,
                                 title=title, author=author,
                                 protein_context=protein_context)


def _generate_docking_report(results: str = None, results_json: str = None,
                             output_path: str = None, title: str = None,
                             author: str = 'Zhuangzhou',
                             protein_context: dict = None,
                             config: dict = None, **kw) -> dict:
    """Generate a docking analysis Word report."""
    from zhuangzhou.report import build_docking_report

    results_dict = _parse_results(results, results_json)
    if isinstance(results_dict, dict) and 'error' in results_dict:
        return results_dict

    return build_docking_report(results_dict, output_path=output_path,
                                title=title, author=author,
                                protein_context=protein_context)


def _generate_md_report(results: str = None, results_json: str = None,
                        output_path: str = None, title: str = None,
                        author: str = 'Zhuangzhou',
                        protein_context: dict = None,
                        config: dict = None, **kw) -> dict:
    """Generate an MD simulation Word report."""
    from zhuangzhou.report import build_md_report

    results_dict = _parse_results(results, results_json)
    if isinstance(results_dict, dict) and 'error' in results_dict:
        return results_dict

    return build_md_report(results_dict, output_path=output_path,
                           title=title, author=author,
                           protein_context=protein_context)


def _generate_qm_report(results: str = None, results_json: str = None,
                        output_path: str = None, title: str = None,
                        author: str = 'Zhuangzhou',
                        config: dict = None, **kw) -> dict:
    """Generate a QM computation Word report (DFT/TS/frequency)."""
    from zhuangzhou.report import build_qm_report

    results_dict = _parse_results(results, results_json)
    if isinstance(results_dict, dict) and 'error' in results_dict:
        return results_dict

    return build_qm_report(results_dict, output_path=output_path,
                           title=title, author=author)


def _generate_general_report(sections: str = None, sections_json: str = None,
                             output_path: str = None, title: str = 'Report',
                             author: str = 'Zhuangzhou',
                             config: dict = None, **kw) -> dict:
    """Generate a free-form Word report from section definitions.

    Accepts sections as:
      - Python list of dicts (direct)
      - JSON string: either a list [...] or a dict {"sections": [...]}
      - File path to a JSON file
      - Via sections_json param (file path or inline JSON string)
      - Via **kw['results'] as fallback (LLM sometimes puts sections there)
    """
    from zhuangzhou.report import build_general_report

    section_list = None

    if isinstance(sections, list):
        section_list = sections
    elif isinstance(sections, dict):
        # LLM may pass a dict with 'sections' key directly
        section_list = sections.get('sections', [sections])
    elif isinstance(sections, str):
        if os.path.exists(sections):
            with open(sections, encoding='utf-8', errors='replace') as f:
                section_list = json.load(f)
        else:
            try:
                parsed = json.loads(sections)
                # Handle {"sections": [...]} wrapper that LLM often produces
                if isinstance(parsed, dict) and 'sections' in parsed:
                    section_list = parsed['sections']
                elif isinstance(parsed, list):
                    section_list = parsed
                elif isinstance(parsed, dict):
                    # Single section dict — wrap in list
                    section_list = [parsed]
                else:
                    section_list = parsed
            except (json.JSONDecodeError, TypeError):
                return {'error': 'Cannot parse sections',
                        'hint': 'Provide a JSON array of section objects'}

    # Try sections_json (file path or inline JSON)
    if section_list is None and sections_json:
        if os.path.exists(sections_json):
            with open(sections_json, encoding='utf-8', errors='replace') as f:
                section_list = json.load(f)
        else:
            # Fallback: try parsing as inline JSON string
            try:
                parsed = json.loads(sections_json)
                if isinstance(parsed, dict) and 'sections' in parsed:
                    section_list = parsed['sections']
                elif isinstance(parsed, list):
                    section_list = parsed
                else:
                    section_list = [parsed] if isinstance(parsed, dict) else None
            except (json.JSONDecodeError, TypeError):
                return {'error': f'sections_json is not a valid file path or JSON string',
                        'hint': 'Pass a file path or a JSON array string'}

    # Fallback: extract from results (LLM sometimes puts sections in results param)
    if section_list is None:
        results = kw.get('results')
        if isinstance(results, str):
            try:
                parsed = json.loads(results)
                if isinstance(parsed, dict) and 'sections' in parsed:
                    section_list = parsed['sections']
                    if not title or title == 'Report':
                        title = parsed.get('title', parsed.get('report_title', title))
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(results, dict) and 'sections' in results:
            section_list = results['sections']
            if not title or title == 'Report':
                title = results.get('title', results.get('report_title', title))

    if not section_list:
        return {'error': 'No sections provided',
                'hint': 'Pass sections list, sections_json file path, or results dict with "sections" key'}

    # Ensure section_list is a list (unwrap nested dict if needed)
    if isinstance(section_list, dict) and 'sections' in section_list:
        section_list = section_list['sections']

    if not output_path:
        output_path = os.path.join(os.getcwd(), 'report.docx')

    return build_general_report(section_list, output_path=output_path,
                                title=title, author=author)


def _run_full_pipeline_report(pdb_id: str = None, pdb_path: str = None,
                              ligand_smiles: str = None, ligand_name: str = None,
                              mutations: str = None,
                              run_docking: bool = True,
                              run_plip: bool = True,
                              run_md: bool = False,
                              output_dir: str = None,
                              author: str = 'Zhuangzhou',
                              config: dict = None, **kw) -> dict:
    """
    Full pipeline: fetch structure → mutate → dock → analyze → write Word report.

    This is the "one natural language command → Word paper" skill.

    Args:
        pdb_id: PDB ID to auto-download (e.g. '1TQN')
        pdb_path: Local PDB file path (alternative to pdb_id)
        ligand_smiles: Ligand SMILES (for conformer generation)
        ligand_name: Ligand common name
        mutations: Comma-separated mutations (e.g. 'V21L,F87A')
        run_docking: Whether to run molecular docking
        run_plip: Whether to run interaction analysis
        run_md: Whether to run MD simulation (slow, ~30min)
        output_dir: Output directory
        author: Report author
    """
    import tempfile
    from zhuangzhou.skills import get_registry

    registry = get_registry(config)
    collected = {
        'protein_name': pdb_id or os.path.splitext(os.path.basename(pdb_path or ''))[0],
        'figures': {},
    }

    if not output_dir:
        output_dir = os.path.join(
            os.getcwd(),
            f'{collected["protein_name"]}_report'
        )
    os.makedirs(output_dir, exist_ok=True)
    collected['output_dir'] = output_dir

    # ── Step 1: Get protein structure ──
    if pdb_id and not pdb_path:
        logger.info("[1] Fetching PDB structure: %s", pdb_id)
        # Try PDB download
        pdb_skill = registry.get('search_pdb_structure')
        if pdb_skill:
            pdb_result = pdb_skill.execute(query=pdb_id, config=config)
            if isinstance(pdb_result, dict) and pdb_result.get('pdb_path'):
                pdb_path = pdb_result['pdb_path']
            elif isinstance(pdb_result, dict) and pdb_result.get('pdb_id'):
                # Download via urllib
                pdb_path = _download_pdb(pdb_id, output_dir)
        if not pdb_path:
            pdb_path = _download_pdb(pdb_id, output_dir)

        if not pdb_path:
            return {'error': f'Failed to fetch PDB: {pdb_id}',
                    'hint': 'Check PDB ID or provide pdb_path'}
        collected['pdb_path'] = pdb_path

    if pdb_path:
        collected['pdb_path'] = pdb_path

    if not pdb_path or not os.path.exists(pdb_path):
        return {'error': 'No PDB structure available',
                'hint': 'Provide pdb_id or pdb_path'}

    # Get protein info
    protein_info_skill = registry.get('get_uniprot_full')
    if protein_info_skill and pdb_id:
        info = protein_info_skill.execute(accession=pdb_id, config=config)
        if isinstance(info, dict) and 'error' not in info:
            collected['protein_info'] = info

    # ── Step 2: Cofactor detection ──
    logger.info("[2] Detecting cofactors...")
    cof_skill = registry.get('cofactor_detection')
    if cof_skill:
        cof_result = cof_skill.execute(pdb_path=pdb_path, config=config)
        if isinstance(cof_result, dict) and 'error' not in cof_result:
            collected['cofactors'] = cof_result.get('cofactors', [])

    # ── Step 3: Mutation analysis ──
    if mutations:
        logger.info("[3] Analyzing mutations: %s", mutations)
        if isinstance(mutations, str):
            mutation_list = [m.strip() for m in mutations.split(',')]
        else:
            mutation_list = list(mutations)

        analyzed_mutations = []
        chem_skill = registry.get('chemical_analysis')
        for mut in mutation_list:
            if chem_skill:
                r = chem_skill.execute(mutation=mut, config=config)
                if isinstance(r, dict) and 'error' not in r:
                    r['mutation'] = mut
                    analyzed_mutations.append(r)
                else:
                    analyzed_mutations.append({'mutation': mut, 'score': 0, 'recommendation': 'Unknown'})
            else:
                analyzed_mutations.append({'mutation': mut, 'score': 0, 'recommendation': 'N/A'})

        collected['mutations'] = analyzed_mutations

        # Mutant structures
        mutate_skill = registry.get('mutate_minimize')
        if mutate_skill:
            logger.info("[3b] Building mutant structures...")
            structures = {}
            for mut in mutation_list[:5]:  # Limit to top 5
                r = mutate_skill.execute(
                    pdb_path=pdb_path, mutations=mut,
                    output_dir=os.path.join(output_dir, 'mutants'),
                    config=config,
                )
                if isinstance(r, dict) and r.get('output_pdb'):
                    structures[mut] = r['output_pdb']
            collected['structures'] = structures

    # ── Step 4: Docking ──
    if run_docking:
        logger.info("[4] Running molecular docking...")
        dock_skill = registry.get('smart_docking') or registry.get('vina_docking')
        if dock_skill:
            dock_params = {'protein_pdb': pdb_path, 'output_dir': output_dir, 'config': config}
            if ligand_smiles:
                # Generate conformer from SMILES
                conf_skill = registry.get('conformer_generate')
                if conf_skill:
                    conf_r = conf_skill.execute(
                        smiles=ligand_smiles, output_dir=output_dir, config=config)
                    if isinstance(conf_r, dict) and conf_r.get('sdf_path'):
                        dock_params['ligand_sdf'] = conf_r['sdf_path']
            dock_r = dock_skill.execute(**dock_params)
            if isinstance(dock_r, dict):
                collected['docking'] = dock_r.get('vina', dock_r)
                collected['vina'] = dock_r.get('vina', dock_r)

    # ── Step 5: PLIP interaction analysis ──
    if run_plip and collected.get('docking'):
        logger.info("[5] Running interaction analysis...")
        complexes = collected.get('docking', {}).get('complexes', [])
        if complexes:
            best_complex = complexes[0].get('path', '')
            plip_skill = registry.get('plip_analysis')
            if plip_skill and best_complex and os.path.exists(best_complex):
                plip_r = plip_skill.execute(
                    complex_pdb=best_complex,
                    output_dir=os.path.join(output_dir, 'plip'),
                    config=config,
                )
                if isinstance(plip_r, dict) and 'error' not in plip_r:
                    collected['interactions'] = plip_r

                # Visualization
                viz_skill = registry.get('plip_visualize')
                if viz_skill:
                    viz_r = viz_skill.execute(
                        complex_pdb=best_complex,
                        plip_results=plip_r,
                        output_dir=os.path.join(output_dir, 'figures'),
                        config=config,
                    )
                    if isinstance(viz_r, dict):
                        for key in ('detail_png', 'global_png', 'detail', 'global'):
                            if viz_r.get(key) and os.path.exists(viz_r[key]):
                                fig_key = 'plip_detail' if 'detail' in key else 'plip_global'
                                collected['figures'][fig_key] = viz_r[key]

    # ── Step 6: Structure visualization ──
    logger.info("[6] Generating structure visualization...")
    render_skill = registry.get('chimerax_render') or registry.get('pymol_view')
    if render_skill:
        render_r = render_skill.execute(
            pdb_path=pdb_path,
            output_dir=os.path.join(output_dir, 'figures'),
            config=config,
        )
        if isinstance(render_r, dict) and render_r.get('png_path'):
            collected['figures']['structure'] = render_r['png_path']

    # ── Step 7: MD Simulation (optional, slow) ──
    if run_md:
        logger.info("[7] Running MD simulation (this may take a while)...")
        md_skill = registry.get('auto_md')
        if md_skill:
            md_r = md_skill.execute(
                pdb_path=pdb_path,
                output_dir=os.path.join(output_dir, 'md'),
                sim_time_ns=1,  # Short for report demo
                config=config,
            )
            if isinstance(md_r, dict) and 'error' not in md_r:
                collected['md_analysis'] = md_r.get('analysis', md_r)
                # Collect MD figures
                for key in ('rmsd_png', 'rmsf_png', 'rg_png', 'hbond_png', 'fel_png'):
                    fig = md_r.get(key)
                    if fig and os.path.exists(fig):
                        collected['figures'][key.replace('_png', '')] = fig

    # ── Step 8: Generate Word report ──
    logger.info("[8] Writing Word report...")
    from zhuangzhou.report import build_mutation_report
    report_path = os.path.join(output_dir, f'{collected["protein_name"]}_report.docx')
    report_result = build_mutation_report(
        collected, output_path=report_path, author=author,
    )

    # Also save collected data as JSON
    json_path = os.path.join(output_dir, f'{collected["protein_name"]}_data.json')
    _save_json_safe(collected, json_path)

    report_result['json_path'] = json_path
    report_result['output_dir'] = output_dir
    return report_result


def _parse_results(results, results_json):
    """Parse results from various input formats."""
    if isinstance(results, dict):
        return results
    if isinstance(results, str):
        if os.path.exists(results):
            with open(results, encoding='utf-8', errors='replace') as f:
                return json.load(f)
        try:
            return json.loads(results)
        except (json.JSONDecodeError, TypeError):
            return {'error': f'Cannot parse results: {str(results)[:200]}',
                    'hint': 'Provide a valid JSON file path or dict'}
    if results_json and os.path.exists(results_json):
        with open(results_json, encoding='utf-8', errors='replace') as f:
            return json.load(f)
    return {'error': 'No results provided', 'hint': 'Pass results dict or file path'}


def _download_pdb(pdb_id: str, output_dir: str) -> str:
    """Download PDB file from RCSB."""
    import urllib.request
    pdb_id = pdb_id.upper().strip()
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    output_path = os.path.join(output_dir, f'{pdb_id}.pdb')
    try:
        urllib.request.urlretrieve(url, output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            logger.info("Downloaded PDB: %s", output_path)
            return output_path
    except Exception as e:
        logger.warning("PDB download failed for %s: %s", pdb_id, e)
    return ''


def _save_json_safe(data: dict, path: str):
    """Save dict to JSON, skipping non-serializable values."""
    def default(o):
        if hasattr(o, '__dict__'):
            return str(o)
        return f'<{type(o).__name__}>'

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=default)
    except Exception as e:
        logger.warning("Failed to save JSON: %s", e)


def _enriched_docking_report_skill(results=None, results_json: str = None,
                                    uniprot_id: str = None,
                                    ligand_smiles: str = None,
                                    output_path: str = None,
                                    title: str = None,
                                    author: str = 'Zhuangzhou',
                                    config: dict = None, **kw) -> dict:
    """
    Skill wrapper for build_enriched_docking_report.

    Accepts docking results as a dict, JSON string, or file path.
    Fetches UniProt/KEGG data, classifies ligand role, checks active site
    consistency, and writes an enriched academic Word report.
    """
    from zhuangzhou.report import build_enriched_docking_report

    results_dict = _parse_results(results, results_json)
    if isinstance(results_dict, dict) and 'error' in results_dict:
        return results_dict

    return build_enriched_docking_report(
        results_dict,
        output_path=output_path,
        title=title,
        author=author,
        uniprot_id=uniprot_id,
        ligand_smiles=ligand_smiles,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Skill Registration
# ══════════════════════════════════════════════════════════════════════════════

def register_skills(registry: SkillRegistry):
    """Register report generation skills."""

    registry.register(Skill(
        name='generate_mutation_report',
        description='Generate a Word (.docx) mutation analysis report from pipeline results',
        category='report',
        fn=_generate_mutation_report,
        inputs=[
            SkillParam('results', 'json', 'Pipeline results dict or JSON string', required=False),
            SkillParam('results_json', 'path', 'Path to results JSON file', required=False),
            SkillParam('output_path', 'path', 'Output .docx file path', required=False),
            SkillParam('title', 'string', 'Report title', required=False),
            SkillParam('author', 'string', 'Author name', required=False, default='Zhuangzhou'),
        ],
        outputs=[
            SkillParam('docx_path', 'path', 'Generated Word document path'),
            SkillParam('figures', 'int', 'Number of figures in report'),
            SkillParam('tables', 'int', 'Number of tables in report'),
        ],
        source='zhuangzhou/report/templates.py',
        estimated_time='~5s',
        output_type='report',
    ))

    registry.register(Skill(
        name='generate_docking_report',
        description='Generate a Word (.docx) molecular docking report',
        category='report',
        fn=_generate_docking_report,
        inputs=[
            SkillParam('results', 'json', 'SmartDocking results dict or JSON string', required=False),
            SkillParam('results_json', 'path', 'Path to docking results JSON file', required=False),
            SkillParam('output_path', 'path', 'Output .docx file path', required=False),
            SkillParam('title', 'string', 'Report title', required=False),
            SkillParam('author', 'string', 'Author name', required=False, default='Zhuangzhou'),
        ],
        outputs=[
            SkillParam('docx_path', 'path', 'Generated Word document path'),
        ],
        source='zhuangzhou/report/templates.py',
        estimated_time='~5s',
        output_type='report',
    ))

    registry.register(Skill(
        name='generate_md_report',
        description='Generate a Word (.docx) molecular dynamics simulation report',
        category='report',
        fn=_generate_md_report,
        inputs=[
            SkillParam('results', 'json', 'MD analysis results dict or JSON string', required=False),
            SkillParam('results_json', 'path', 'Path to MD results JSON file', required=False),
            SkillParam('output_path', 'path', 'Output .docx file path', required=False),
            SkillParam('title', 'string', 'Report title', required=False),
            SkillParam('author', 'string', 'Author name', required=False, default='Zhuangzhou'),
        ],
        outputs=[
            SkillParam('docx_path', 'path', 'Generated Word document path'),
        ],
        source='zhuangzhou/report/templates.py',
        estimated_time='~5s',
    ))

    registry.register(Skill(
        name='generate_qm_report',
        description=(
            'Generate a publication-quality QM computation Word report (.docx) '
            'covering DFT geometry optimization, transition state characterisation, '
            'frequency analysis, reaction energy profile, and mechanistic discussion.'
        ),
        category='report',
        fn=_generate_qm_report,
        inputs=[
            SkillParam('results', 'json', 'QM pipeline results dict or JSON string', required=False),
            SkillParam('results_json', 'path', 'Path to QM results JSON file', required=False),
            SkillParam('output_path', 'path', 'Output .docx file path', required=False),
            SkillParam('title', 'string', 'Report title', required=False),
            SkillParam('author', 'string', 'Author name', required=False, default='Zhuangzhou'),
        ],
        outputs=[
            SkillParam('docx_path', 'path', 'Generated Word document path'),
        ],
        source='zhuangzhou/report/templates.py',
        estimated_time='~5s',
    ))

    registry.register(Skill(
        name='generate_general_report',
        description='Generate a free-form Word (.docx) report from section definitions (LLM-driven)',
        category='report',
        fn=_generate_general_report,
        inputs=[
            SkillParam('sections', 'json', 'List of section dicts [{type, text/data, ...}]', required=False),
            SkillParam('sections_json', 'path', 'Path to sections JSON file', required=False),
            SkillParam('output_path', 'path', 'Output .docx file path', required=False),
            SkillParam('title', 'string', 'Report title', required=False, default='Report'),
            SkillParam('author', 'string', 'Author name', required=False, default='Zhuangzhou'),
        ],
        outputs=[
            SkillParam('docx_path', 'path', 'Generated Word document path'),
        ],
        source='zhuangzhou/report/templates.py',
        estimated_time='~5s',
    ))

    registry.register(Skill(
        name='full_pipeline_report',
        description='Full automated pipeline: fetch protein → mutate → dock → analyze → generate Word report. One command, complete paper.',
        category='report',
        fn=_run_full_pipeline_report,
        inputs=[
            SkillParam('pdb_id', 'string', 'PDB ID to auto-download (e.g. 1TQN)', required=False),
            SkillParam('pdb_path', 'path', 'Local PDB file path', required=False),
            SkillParam('ligand_smiles', 'string', 'Ligand SMILES string', required=False),
            SkillParam('ligand_name', 'string', 'Ligand common name', required=False),
            SkillParam('mutations', 'string', 'Comma-separated mutations (e.g. V21L,F87A)', required=False),
            SkillParam('run_docking', 'bool', 'Run molecular docking', required=False, default=True),
            SkillParam('run_plip', 'bool', 'Run interaction analysis', required=False, default=True),
            SkillParam('run_md', 'bool', 'Run MD simulation (slow)', required=False, default=False),
            SkillParam('output_dir', 'path', 'Output directory (default: CWD/<pdb_stem>_report/)',
                       required=False, is_output_path=True),
            SkillParam('author', 'string', 'Report author', required=False, default='Zhuangzhou'),
        ],
        outputs=[
            SkillParam('docx_path', 'path', 'Generated Word document path'),
            SkillParam('json_path', 'path', 'Raw data JSON path'),
            SkillParam('output_dir', 'path', 'Output directory'),
        ],
        requires_network=True,
        source='zhuangzhou/skills/report.py',
        risk_level='medium',
        estimated_time='~5-30min',
    ))

    registry.register(Skill(
        name='enriched_docking_report',
        description=(
            'Academic-grade docking report with UniProt active site analysis, '
            'KEGG pathway context, ligand role classification '
            '(inhibitor/substrate/activator), active site consistency analysis, '
            'and mechanistic hypothesis. Extends generate_docking_report with '
            'five additional enrichment sections.'
        ),
        category='report',
        fn=_enriched_docking_report_skill,
        inputs=[
            SkillParam('results', 'json',
                       'Docking results dict (same format as generate_docking_report)',
                       required=False),
            SkillParam('results_json', 'path',
                       'Path to docking results JSON file', required=False),
            SkillParam('uniprot_id', 'string',
                       'UniProt accession for the target protein (e.g. P08684 for CYP3A4)',
                       required=False),
            SkillParam('ligand_smiles', 'string',
                       'SMILES string of the docked ligand (improves MW-based classification)',
                       required=False),
            SkillParam('output_path', 'path',
                       'Output .docx file path (auto-generated if not provided)',
                       required=False, is_output_path=True),
            SkillParam('title', 'string', 'Report title', required=False),
            SkillParam('author', 'string', 'Author name', required=False,
                       default='Zhuangzhou'),
        ],
        outputs=[
            SkillParam('docx_path', 'path', 'Generated enriched Word document path'),
            SkillParam('figures', 'int', 'Number of figures in report'),
            SkillParam('tables', 'int', 'Number of tables in report'),
            SkillParam('uniprot_fetched', 'bool', 'Whether UniProt data was successfully fetched'),
            SkillParam('kegg_fetched', 'bool', 'Whether KEGG data was successfully fetched'),
            SkillParam('ligand_role', 'string', 'Predicted ligand role: inhibitor/substrate/activator/unknown'),
            SkillParam('consistency_pct', 'float', 'Active site overlap percentage (PLIP vs UniProt)'),
        ],
        requires_network=True,
        source='zhuangzhou/report/templates.py',
        estimated_time='~10-30s',
        output_type='report',
    ))


def _generate_synbio_report(results: str = None, results_json: str = None,
                            output_path: str = None, title: str = None,
                            author: str = 'Zhuangzhou',
                            config: dict = None, **kw) -> dict:
    """Generate a bilingual synbio design Word report."""
    from zhuangzhou.report.templates import build_synbio_report

    results_dict = _parse_results(results, results_json)
    if isinstance(results_dict, dict) and 'error' in results_dict:
        return results_dict

    return build_synbio_report(results_dict, output_path=output_path,
                               title=title, author=author)
