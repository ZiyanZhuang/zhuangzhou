"""
Zhuangzhou Core Framework
=========================
The overarching enzyme engineering system that integrates:
  - BeTaThink KG + Document KB as the knowledge brain ("hippocampus")
  - ChemVision Box: PLIP, structural analysis, chemical intuition
  - BioPred Box: PLM (ESM-2), Fold prediction, Docking
  - Deep Box: GROMACS, Vina, ORCA computational engines
  - Agent Layer: LLM decision-making with KG-guided reasoning

Architecture (from design diagram):
  ┌──────────────────────────────────────────────────────────────┐
  │  Agent (LLM + Decision Framework)                           │
  │    ├── KG Plus (BeTaThink Neo4j + PostgreSQL)              │
  │    └── RAG (BeTaThink Milvus vector search)                │
  ├──────────────────────────────────────────────────────────────┤
  │  ChemVision Box          │  BioPred Box     │  Deep Box     │
  │  PLIP, Prodigy          │  PLM (ESM-2)     │  ORCA         │
  │  Chemical Intuition     │  EzReac          │  GROMACS      │
  │  Scripts & Analysis     │  Fold, Docking   │  Vina GPU     │
  └──────────────────────────────────────────────────────────────┘
            ↕ feedback loop ↕
         Wet Lab → Data Loop
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)

# BeTaThink integration
BETATHINK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BETATHINK_ROOT)


def compute_epistasis_risk(combo_mutations):
    """Estimate epistasis risk for mutation combinations."""
    positions = [m['position'] for m in combo_mutations]
    roles = [m.get('role', 'unknown') for m in combo_mutations]
    risk_factors = []

    for i in range(len(combo_mutations)):
        for j in range(i + 1, len(combo_mutations)):
            if abs(positions[i] - positions[j]) <= 5:
                risk_factors.append('Adjacent ({}-{})'.format(positions[i], positions[j]))
            if roles[i] == roles[j]:
                risk_factors.append('Same role: {}'.format(roles[i]))

    n_active = sum(1 for r in roles if r in ('active_site', 'ligand_interacting'))
    n_stab = sum(1 for r in roles if r in ('buried_core', 'surface', 'rigid_core'))

    if n_active >= 1 and n_stab >= 1:
        return {'risk_level': 'LOW', 'epistasis_type': 'likely_positive', 'risk_factors': risk_factors}
    elif n_active >= 2:
        return {'risk_level': 'HIGH', 'epistasis_type': 'possible_negative', 'risk_factors': risk_factors}
    elif n_stab >= 2 and all(abs(positions[i] - positions[j]) > 15
                              for i in range(len(positions))
                              for j in range(i+1, len(positions))):
        return {'risk_level': 'LOW', 'epistasis_type': 'likely_additive', 'risk_factors': risk_factors}
    else:
        return {'risk_level': 'MODERATE', 'epistasis_type': 'uncertain', 'risk_factors': risk_factors}


class KnowledgeBrain:
    """
    Zhuangzhou's knowledge hippocampus, backed by BeTaThink storage + AutoGraphRAG.

    Provides:
      - Entity search (PostgreSQL)
      - KG path discovery (Neo4j beam search)
      - Document retrieval (Milvus vector search)
      - Literature context (DOI/chunk lookup)
      - GraphRAG domain knowledge (parquet KGs, 6 domains)

    Degrades gracefully when databases are unavailable (offline mode).
    """

    def __init__(self):
        self._storage = None
        self._service = None
        self._online = False
        self._graphrag = None

    @property
    def graphrag(self):
        if self._graphrag is None:
            from .brain.graphrag import GraphRAGBridge
            self._graphrag = GraphRAGBridge()
        return self._graphrag

    def connect(self) -> bool:
        """Try to connect to BeTaThink storage. Returns True if connected."""
        try:
            from storage.manager import StorageManager
            self._storage = StorageManager()
            self._storage.initialize()
            self._online = True
            logger.info("KnowledgeBrain: Connected to BeTaThink storage")
        except Exception as e:
            logger.warning("KnowledgeBrain: BeTaThink storage unavailable (%s). Using offline mode.", e)
            self._online = False

        # GraphRAG is always available (file-based, no DB needed)
        kg_count = len(self.graphrag.list_kgs())
        logger.info("KnowledgeBrain: %d GraphRAG knowledge graphs available", kg_count)
        return self._online

    @property
    def is_online(self) -> bool:
        return self._online

    def search_entity(self, name: str, entity_type: str = None) -> List[dict]:
        """Search for biological entities in the KG."""
        if not self._online:
            return self._offline_entity_search(name, entity_type)

        try:
            results = self._storage.entities.search_entities_by_name(name)
            if entity_type:
                results = [r for r in results if r.get('type') == entity_type]
            return results
        except Exception as e:
            logger.warning("Entity search failed: %s", e)
            return self._offline_entity_search(name, entity_type)

    def find_kg_paths(self, source_name: str, target_name: str,
                      max_hops: int = 5) -> List[dict]:
        """Find knowledge graph paths between two entities."""
        if not self._online:
            return []

        try:
            from service.search.path_finder import PathFinder
            source = self.search_entity(source_name)
            target = self.search_entity(target_name)
            if not source or not target:
                return []

            finder = PathFinder(self._storage)
            paths = finder.find_paths(
                source[0]['id'], target[0]['id'],
                max_hops=max_hops, beam_width=10
            )
            return paths
        except Exception as e:
            logger.warning("KG path search failed: %s", e)
            return []

    def search_documents(self, query: str, top_k: int = 5) -> List[dict]:
        """Semantic search in document knowledge base."""
        if not self._online or not self._storage.vectors:
            return []

        try:
            from processing.embedding.vectorizer import Vectorizer
            vec = Vectorizer()
            query_vector = vec.embed_text(query)
            results = self._storage.vectors.search_similar_vectors(
                query_vector, top_k=top_k
            )
            # Enrich with chunk text
            enriched = []
            for hit in results:
                chunk_id = hit.get('chunk_id')
                if chunk_id:
                    chunk = self._storage.chunks.get_chunk_by_id(chunk_id)
                    if chunk:
                        enriched.append({
                            'chunk_id': chunk_id,
                            'text': chunk.get('text', ''),
                            'doc_id': chunk.get('doc_id', ''),
                            'section': chunk.get('section', ''),
                            'score': hit.get('score', 0),
                        })
            return enriched
        except Exception as e:
            logger.warning("Document search failed: %s", e)
            return []

    def get_entity_context(self, entity_id: str) -> dict:
        """Get full KG context for an entity (edges, claims, consensus)."""
        if not self._online:
            return {}

        try:
            edges = self._storage.evidence_graph.get_outgoing_edges(entity_id)
            incoming = self._storage.evidence_graph.get_incoming_edges(entity_id)
            return {
                'entity_id': entity_id,
                'outgoing_edges': edges,
                'incoming_edges': incoming,
            }
        except Exception as e:
            logger.warning("Entity context failed: %s", e)
            return {}

    def get_literature_for_entity(self, entity_name: str) -> List[dict]:
        """Find literature citations for an entity."""
        if not self._online:
            return []

        try:
            entities = self.search_entity(entity_name)
            if not entities:
                return []
            citations = self._storage.evidence_graph.get_literature_citations(
                entities[0]['id']
            )
            return citations
        except Exception as e:
            logger.warning("Literature search failed: %s", e)
            return []

    def _offline_entity_search(self, name: str, entity_type: str = None) -> List[dict]:
        """Fallback entity search using built-in knowledge."""
        from .literature_rag import CATALYTIC_MECHANISMS, KNOWN_BENEFICIAL_MUTATIONS
        results = []
        name_lower = name.lower()
        for mech_name, mech in CATALYTIC_MECHANISMS.items():
            if name_lower in mech_name:
                results.append({
                    'name': mech_name,
                    'type': 'Mechanism',
                    'data': mech,
                })
        return results

    def store_mutation_result(self, mutation: str, result: dict) -> bool:
        """Store mutation analysis result back to KG for future retrieval."""
        if not self._online:
            return False
        try:
            entity_data = {
                'id': 'mutation_{}'.format(mutation.replace(' ', '_')),
                'type': 'Mutation',
                'standard_name': mutation,
                'metadata': result,
            }
            self._storage.entities.create_entity(entity_data)
            return True
        except Exception as e:
            logger.warning("Failed to store mutation result: %s", e)
            return False

    def enrich_mutation_with_graphrag(self, mutation_str: str,
                                       enzyme_class: str = '',
                                       protein_name: str = '') -> dict:
        """Enrich a mutation with GraphRAG domain knowledge."""
        try:
            return self.graphrag.get_mutation_evidence(
                mutation_str, enzyme_class, protein_name)
        except Exception as e:
            logger.debug("GraphRAG enrichment failed: %s", e)
            return {'n_hits': 0}


class ChemVisionBox:
    """
    Chemical structure analysis box.

    Integrates: PLIP, Prodigy, structural scripts, chemical intuition.
    """

    def __init__(self, unimol_scripts: str = None):
        self.scripts_dir = unimol_scripts or os.path.join(
            BETATHINK_ROOT, '..', 'Code', 'Uni-Mol-main', 'scripts'
        )

    def run_plip(self, complex_pdb: str) -> dict:
        """Run PLIP interaction analysis (via WSL)."""
        try:
            sys.path.insert(0, self.scripts_dir)
            from plip_analysis import run_plip_analysis
            return run_plip_analysis(complex_pdb)
        except ImportError:
            logger.warning("plip_analysis not available, using mock")
            return {'interactions': [], 'mock': True}
        except Exception as e:
            logger.warning("PLIP failed: %s", e)
            return {'error': str(e), 'interactions': []}

    def analyze_structure(self, pdb_path: str, binding_site: set) -> dict:
        """Comprehensive structural analysis."""
        from .chemical_intuition import AA_PROPERTIES
        from .structural_context import build_residue_context

        # Parse PDB
        residue_coords = {}
        residue_names = {}
        with open(pdb_path, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('ATOM'):
                    resnum = int(line[22:26].strip())
                    atom_name = line[12:16].strip()
                    resname = line[17:20].strip()
                    if atom_name == 'CA':
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        residue_coords[resnum] = (x, y, z)
                        residue_names[resnum] = resname

        # Active site center
        bs_center = None
        if binding_site:
            bs_coords = [residue_coords[r] for r in binding_site if r in residue_coords]
            if bs_coords:
                bs_center = tuple(sum(c[i] for c in bs_coords) / len(bs_coords)
                                 for i in range(3))

        return {
            'n_residues': len(residue_coords),
            'residue_coords': residue_coords,
            'residue_names': residue_names,
            'binding_site_center': bs_center,
            'binding_site': binding_site,
        }

    def classify_mutation(self, position: int, wt_aa: str, mut_aa: str,
                          structure: dict, interactions: dict = None,
                          is_conserved: bool = False) -> dict:
        """Classify a mutation with chemical reasoning."""
        from .chemical_intuition import analyze_mutation_chemistry

        coords = structure.get('residue_coords', {})
        bs_center = structure.get('binding_site_center')
        binding_site = structure.get('binding_site', set())

        # Compute distance to active site
        dist = None
        if bs_center and position in coords:
            px, py, pz = coords[position]
            dist = sum((px - bs_center[i]) ** 2 for i in range(3)) ** 0.5

        # Build context
        role = 'active_site' if position in binding_site else (
            'second_shell' if dist and dist < 8 else (
                'surface' if self._is_surface(position, coords) else 'buried_core'
            )
        )

        # Estimate secondary structure from backbone geometry
        ss = self._estimate_secondary_structure(position, coords)

        context = {
            'role': role,
            'secondary_structure': ss,
            'interactions': interactions.get(position, []) if interactions else [],
            'neighbors': self._get_neighbor_types(position, coords, structure.get('residue_names', {})),
            'distance_to_site': dist,
            'sasa': None,
            'in_conserved_motif': is_conserved,
        }

        analysis = analyze_mutation_chemistry(wt_aa, mut_aa, context)
        analysis['role'] = role
        analysis['distance_to_site'] = dist
        return analysis

    def _estimate_secondary_structure(self, position: int, coords: dict) -> str:
        """Estimate secondary structure from CA distances to neighbors."""
        if position not in coords:
            return 'loop'
        # Check if consecutive CAs have helix-like spacing (~5.4A for i to i+3)
        # or sheet-like spacing (~6.5-7A for i to i+2)
        px, py, pz = coords[position]
        has_i3 = position + 3 in coords
        has_i2 = position + 2 in coords
        if has_i3:
            x3, y3, z3 = coords[position + 3]
            d_i3 = ((px - x3)**2 + (py - y3)**2 + (pz - z3)**2) ** 0.5
            if 4.5 < d_i3 < 6.5:
                return 'helix'
        if has_i2:
            x2, y2, z2 = coords[position + 2]
            d_i2 = ((px - x2)**2 + (py - y2)**2 + (pz - z2)**2) ** 0.5
            if 6.0 < d_i2 < 7.5:
                return 'sheet'
        return 'loop'

    def _get_neighbor_types(self, position: int, coords: dict,
                            residue_names: dict) -> list:
        """Get amino acid types of spatial neighbors within 8A."""
        if position not in coords:
            return []
        px, py, pz = coords[position]
        neighbors = []
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }
        for pos2, (x2, y2, z2) in coords.items():
            if pos2 == position:
                continue
            d = ((px - x2)**2 + (py - y2)**2 + (pz - z2)**2) ** 0.5
            if d < 8:
                rn = residue_names.get(pos2, '')
                aa = three_to_one.get(rn, '?')
                neighbors.append(aa)
        return neighbors

    def _is_surface(self, position: int, coords: dict) -> bool:
        """Simple surface detection by neighbor count."""
        if position not in coords:
            return False
        px, py, pz = coords[position]
        n_contacts = 0
        for pos2, (x2, y2, z2) in coords.items():
            if pos2 == position:
                continue
            d = ((px - x2)**2 + (py - y2)**2 + (pz - z2)**2) ** 0.5
            if d < 10:
                n_contacts += 1
        return n_contacts < 15  # Few contacts = surface


class BioPredBox:
    """
    Biological prediction box.

    Integrates: ESM-2 (PLM), EzReac, AlphaFold/ESMFold, Docking.
    Delegates to zhuangzhou.tools for actual computation.
    """

    def __init__(self):
        pass

    def load_esm2_results(self, result_dirs: List[str]) -> List[dict]:
        """Load and merge ESM-2 multi-model results."""
        from .tools import load_esm2_results
        return load_esm2_results(result_dirs)

    def load_docking_results(self, docking_dir: str) -> Optional[dict]:
        """Load batch docking results."""
        path = os.path.join(docking_dir, 'batch_results.json')
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8', errors='replace') as f:
            return json.load(f)


class DeepBox:
    """
    Deep computational box.

    Wraps: GROMACS, Vina, ORCA, LigBuilder, etc.
    Currently delegates to BeTaThink's bio_tools (async) or scripts.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.vina_path = self.config.get(
            'vina_path', 'E:/computing bio/MDS-gmx/Docking/vina/vina.exe')
        self.gmx_path = self.config.get(
            'gmx_path', 'E:/gmx2025.1_GPU/bin/gmx.exe')

    def check_available_tools(self) -> dict:
        """Check which computational tools are available."""
        import shutil
        return {
            'vina': os.path.exists(self.vina_path),
            'gromacs': os.path.exists(self.gmx_path),
            'orca': shutil.which('orca') is not None,
            'wsl': shutil.which('wsl') is not None,
        }


class ZhuangzhouFramework:
    """
    The overarching Zhuangzhou framework.

    Orchestrates the full enzyme engineering pipeline:
      Phase 1: LLM+KG global screening (conservation, lethal mutations)
      Phase 2: Structure reconstruction (Track A: Core, Track B: Shell)
      Phase 3: Combination assembly + epistasis screening
      Phase 4: Dynamic mechanism validation (MD)
      Phase 5: Wet-lab feedback loop

    Uses adaptive weighting: first-principles for conserved sites,
    PLM for non-conserved regions.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.brain = KnowledgeBrain()
        self.chemvision = ChemVisionBox(
            self.config.get('scripts_dir'))
        self.biopred = BioPredBox()
        self.deepbox = DeepBox(self.config)

        # State
        self._initialized = False
        self._results = {}

    def initialize(self) -> dict:
        """Initialize all components."""
        status = {
            'brain_online': self.brain.connect(),
            'tools': self.deepbox.check_available_tools(),
            'timestamp': datetime.now().isoformat(),
        }
        self._initialized = True
        logger.info("Zhuangzhou initialized: brain=%s tools=%s",
                     status['brain_online'], status['tools'])
        return status

    def run_enzyme_engineering(
        self,
        pdb_path: str,
        binding_site: Set[int],
        esm2_dirs: List[str],
        docking_dir: str = None,
        enzyme_class: str = '',
        ligand: str = '',
        protein_name: str = '',
        sequence: str = '',
        conserved_positions: Set[int] = None,
        plip_json: str = None,
        fpocket_json: str = None,
        output_dir: str = None,
    ) -> dict:
        """
        Run the complete 5-phase Zhuangzhou enzyme engineering pipeline.
        """
        from zhuangzhou.ui.core import ZZInterface, ZZStatus
        import time

        if not output_dir:
            stem = os.path.splitext(os.path.basename(pdb_path))[0] if pdb_path else 'zhuangzhou'
            output_dir = os.path.join(os.getcwd(), f'{stem}_pipeline')
        os.makedirs(output_dir, exist_ok=True)
        conserved = conserved_positions or set()
        results = {'phases': {}, 'timestamp': datetime.now().isoformat()}

        ZZInterface.render_logo()
        
        start_time = time.perf_counter()
        with ZZStatus(start_time, action="PIPELINE_START") as status:
            if not self._initialized:
                self.initialize()

            # ═══ Phase 1: Global Screening (LLM+KG) ═══
            status.update(action="PHASE 1: SCREENING")
            ZZInterface.print_action("Phase 1: Global Screening (PLM + Conservation + KG)")
            t1 = time.time()
            phase1 = self._phase1_global_screening(
                esm2_dirs, binding_site, conserved, enzyme_class
            )
            results['phases']['phase1_screening'] = phase1
            ZZInterface.print_action("Phase 1 Complete", "DONE", time.time() - t1)

            # ═══ Phase 2: Structural Reconstruction ═══
            status.update(action="PHASE 2: STRUCTURE")
            ZZInterface.print_action("Phase 2: Structural Reconstruction")
            t2 = time.time()
            phase2 = self._phase2_structural_analysis(
                pdb_path, phase1['mutations'], binding_site, conserved,
                plip_json, enzyme_class
            )
            results['phases']['phase2_structure'] = phase2
            ZZInterface.print_action("Phase 2 Complete", "DONE", time.time() - t2)

            # ═══ Phase 3: Combination Assembly ═══
            status.update(action="PHASE 3: COMBINATION")
            ZZInterface.print_action("Phase 3: Combination Assembly + Epistasis")
            t3 = time.time()
            phase3 = self._phase3_combination_design(
                phase2['scored_mutations'], binding_site, docking_dir
            )
            results['phases']['phase3_combinations'] = phase3
            ZZInterface.print_action("Phase 3 Complete", "DONE", time.time() - t3)

            # ═══ Phase 4: Validation Recommendations ═══
            status.update(action="PHASE 4: VALIDATION")
            ZZInterface.print_action("Phase 4: Validation Recommendations")
            t4 = time.time()
            phase4 = self._phase4_validation_plan(
                phase2['scored_mutations'], phase3['combinations']
            )
            results['phases']['phase4_validation'] = phase4
            ZZInterface.print_action("Phase 4 Complete", "DONE", time.time() - t4)

            # ═══ Phase 5: Output ═══
            status.update(action="PHASE 5: REPORT")
            ZZInterface.print_action("Phase 5: Generating Reports")
            report = self._generate_full_report(
                results, protein_name, enzyme_class, ligand,
                binding_site, conserved, output_dir
            )
            results['report_path'] = os.path.join(output_dir, 'zhuangzhou_full_report.md')
            results['json_path'] = os.path.join(output_dir, 'zhuangzhou_results.json')

            # Save JSON
            with open(results['json_path'], 'w', encoding='utf-8') as f:
                # Clean for serialization
                clean = self._clean_for_json(results)
                json.dump(clean, f, indent=2, ensure_ascii=False, default=str)
            
            ZZInterface.print_action("Phase 5 Complete", "DONE")

        print('\n' + '=' * 70)
        print('RESULTS SUMMARY')
        print('=' * 70)
        self._print_summary(phase2['scored_mutations'], phase3['combinations'])

        return results

    def _phase1_global_screening(self, esm2_dirs, binding_site, conserved,
                                  enzyme_class):
        """Phase 1: Load PLM results, enrich with KG, filter lethal."""
        mutations = self.biopred.load_esm2_results(esm2_dirs)

        # KG enrichment: search for known mutation effects
        n_enriched = 0
        n_graphrag = 0
        for m in mutations:
            mut_str = m['mutation']

            # Search KG for known mutation phenotypes
            if self.brain.is_online:
                results = self.brain.search_entity(mut_str)
                if results:
                    m['kg_annotations'] = results
                    n_enriched += 1

                # Also search for enzyme class context
                if enzyme_class:
                    docs = self.brain.search_documents(
                        '{} {} mutation effect'.format(enzyme_class, mut_str),
                        top_k=2
                    )
                    if docs:
                        m['literature_hits'] = docs
                        n_enriched += 1

            # GraphRAG enrichment (always available, file-based)
            graphrag_ctx = self.brain.enrich_mutation_with_graphrag(
                mut_str, enzyme_class)
            if graphrag_ctx.get('n_hits', 0) > 0:
                m['graphrag_context'] = graphrag_ctx
                n_graphrag += 1

            # Conservation check: flag mutations at conserved catalytic residues
            if m['position'] in conserved and m['position'] in binding_site:
                m['conservation_flag'] = 'CONSERVED_ACTIVE_SITE'
                m['plm_trust'] = 'LOW'
            elif m['position'] in conserved:
                m['conservation_flag'] = 'CONSERVED'
                m['plm_trust'] = 'MEDIUM'
            else:
                m['conservation_flag'] = 'VARIABLE'
                m['plm_trust'] = 'HIGH'

        return {
            'n_mutations': len(mutations),
            'n_kg_enriched': n_enriched,
            'n_graphrag_enriched': n_graphrag,
            'mutations': mutations,
        }

    def _phase2_structural_analysis(self, pdb_path, mutations, binding_site,
                                     conserved, plip_json, enzyme_class):
        """Phase 2: Structural analysis with Track A (Core) and Track B (Shell)."""
        structure = self.chemvision.analyze_structure(pdb_path, binding_site)

        # PLIP interactions
        interactions = {}
        if plip_json and os.path.exists(plip_json):
            from .structural_context import parse_plip_interactions
            interactions = parse_plip_interactions(plip_json)

        # Score each mutation
        scored = []
        for m in mutations:
            chem = self.chemvision.classify_mutation(
                m['position'], m['wt'], m['mut'],
                structure, interactions,
                is_conserved=(m['position'] in conserved),
            )

            # Determine track: Core (A) = in binding site OR within 8A
            dist = chem.get('distance_to_site')
            if m['position'] in binding_site:
                track = 'A'  # Always core if in binding site
            elif dist is not None and dist <= 8:
                track = 'A'  # Core: within 8A of active site
            else:
                track = 'B'  # Shell: channels, surface, distal

            # Adaptive fusion
            if m.get('conservation_flag') == 'CONSERVED_ACTIVE_SITE':
                w_chem, w_plm = 0.80, 0.20
            elif m.get('plm_trust') == 'HIGH':
                w_chem, w_plm = 0.20, 0.80
            else:
                w_chem, w_plm = 0.50, 0.50

            # Normalize and fuse
            plm_score = m.get('consensus_score', m.get('delta_ll', 0))
            chem_score = chem.get('chemical_score', 0)

            # Simple fusion: scale chemical to comparable range
            plm_norm = max(0, min(1, (plm_score + 2) / 7))  # -2 to 5 → 0 to 1
            chem_norm = max(0, min(1, (chem_score + 5) / 10))  # -5 to 5 → 0 to 1

            zz_score = w_chem * chem_norm + w_plm * plm_norm

            scored.append({
                **m,
                'track': track,
                'role': chem.get('role', 'unknown'),
                'chemical_score': chem_score,
                'zz_score': round(zz_score, 4),
                'w_chem': w_chem,
                'w_plm': w_plm,
                'distance_to_site': dist,
                'recommendation': chem.get('recommendation', ''),
                'reasoning': chem.get('reasoning', []),
                'warnings': chem.get('warnings', []),
                'opportunities': chem.get('opportunities', []),
            })

        scored.sort(key=lambda x: -x['zz_score'])

        track_a = [m for m in scored if m['track'] == 'A']
        track_b = [m for m in scored if m['track'] == 'B']

        return {
            'scored_mutations': scored,
            'track_a': track_a,
            'track_b': track_b,
            'track_a_count': len(track_a),
            'track_b_count': len(track_b),
            'structure': {
                'n_residues': structure['n_residues'],
                'binding_site_center': structure['binding_site_center'],
            },
        }

    def _phase3_combination_design(self, scored_mutations, binding_site,
                                    docking_dir):
        """Phase 3: Design multi-mutation combinations with epistasis."""
        # compute_epistasis_risk is defined at module level

        # Load docking if available
        docking = self.biopred.load_docking_results(docking_dir) if docking_dir else None

        # Best per position
        by_pos = {}
        for m in scored_mutations:
            pos = m['position']
            if pos not in by_pos or m['zz_score'] > by_pos[pos]['zz_score']:
                by_pos[pos] = m
        unique = sorted(by_pos.values(), key=lambda x: -x['zz_score'])

        # Separate by track
        track_a_best = [m for m in unique if m['track'] == 'A'][:5]
        track_b_best = [m for m in unique if m['track'] == 'B'][:8]

        combinations = []

        # Strategy 1: Track A + Track B orthogonal
        from itertools import product as iter_product, combinations as iter_combos
        for a in track_a_best[:3]:
            for b in track_b_best[:5]:
                if a['position'] == b['position']:
                    continue
                epi = compute_epistasis_risk([a, b])
                combined = a['zz_score'] + b['zz_score']
                penalty = {'LOW': 1.0, 'MODERATE': 0.85, 'HIGH': 0.7}
                adj = combined * penalty.get(epi['risk_level'], 0.85)

                combinations.append({
                    'mutations': [a['mutation'], b['mutation']],
                    'positions': [a['position'], b['position']],
                    'tracks': [a['track'], b['track']],
                    'roles': [a['role'], b['role']],
                    'zz_combined': round(combined, 4),
                    'zz_adjusted': round(adj, 4),
                    'epistasis': epi,
                    'strategy': 'core_shell' if a['track'] == 'A' else 'shell_pair',
                })

        # Strategy 2: Triple (1 core + 2 shell)
        for a in track_a_best[:2]:
            for b1, b2 in iter_combos(track_b_best[:5], 2):
                if len({a['position'], b1['position'], b2['position']}) < 3:
                    continue
                combo = [a, b1, b2]
                epi = compute_epistasis_risk(combo)
                combined = sum(m['zz_score'] for m in combo)
                penalty = {'LOW': 1.0, 'MODERATE': 0.85, 'HIGH': 0.7}
                adj = combined * penalty.get(epi['risk_level'], 0.85)

                combinations.append({
                    'mutations': [m['mutation'] for m in combo],
                    'positions': [m['position'] for m in combo],
                    'tracks': [m['track'] for m in combo],
                    'roles': [m['role'] for m in combo],
                    'zz_combined': round(combined, 4),
                    'zz_adjusted': round(adj, 4),
                    'epistasis': epi,
                    'strategy': 'core_2shell',
                })

        # Strategy 3: Stability stacking (shell only, far apart)
        for b1, b2 in iter_combos(track_b_best[:6], 2):
            if b1['position'] == b2['position']:
                continue
            if abs(b1['position'] - b2['position']) < 5:
                continue
            epi = compute_epistasis_risk([b1, b2])
            combined = b1['zz_score'] + b2['zz_score']
            penalty = {'LOW': 1.0, 'MODERATE': 0.85, 'HIGH': 0.7}
            adj = combined * penalty.get(epi['risk_level'], 0.85)
            combinations.append({
                'mutations': [b1['mutation'], b2['mutation']],
                'positions': [b1['position'], b2['position']],
                'tracks': ['B', 'B'],
                'roles': [b1['role'], b2['role']],
                'zz_combined': round(combined, 4),
                'zz_adjusted': round(adj, 4),
                'epistasis': epi,
                'strategy': 'stability_stack',
            })

        # Sort and deduplicate
        combinations.sort(key=lambda x: -x['zz_adjusted'])
        seen = set()
        unique_combos = []
        for c in combinations:
            key = tuple(sorted(c['positions']))
            if key not in seen:
                seen.add(key)
                unique_combos.append(c)

        return {'combinations': unique_combos[:30]}

    def _phase4_validation_plan(self, scored_mutations, combinations):
        """Phase 4: Generate experimental validation plan."""
        tier1 = scored_mutations[:5]
        tier2_combos = [c for c in combinations if len(c['mutations']) == 2][:5]
        tier3_combos = [c for c in combinations if len(c['mutations']) >= 3][:3]

        # MD priorities
        md_candidates = [m for m in scored_mutations[:10]
                        if m.get('track') == 'A' or m.get('role') == 'active_site']

        return {
            'tier1_singles': [m['mutation'] for m in tier1],
            'tier2_doubles': [' + '.join(c['mutations']) for c in tier2_combos],
            'tier3_triples': [' + '.join(c['mutations']) for c in tier3_combos],
            'md_candidates': [m['mutation'] for m in md_candidates[:3]],
            'total_experiments': len(tier1) + len(tier2_combos) + len(tier3_combos),
        }

    def _generate_full_report(self, results, protein_name, enzyme_class, ligand,
                               binding_site, conserved, output_dir):
        """Generate the comprehensive Markdown report."""
        p1 = results['phases'].get('phase1_screening', {})
        p2 = results['phases'].get('phase2_structure', {})
        p3 = results['phases'].get('phase3_combinations', {})
        p4 = results['phases'].get('phase4_validation', {})

        scored = p2.get('scored_mutations', [])
        combos = p3.get('combinations', [])

        lines = []
        lines.append('# Zhuangzhou Enzyme Engineering Report')
        lines.append('')
        lines.append('> Framework: Zhuangzhou v0.2 | BeTaThink KG Integration')
        lines.append('> Generated: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M')))
        lines.append('')
        lines.append('## System Overview')
        lines.append('')
        lines.append('| Property | Value |')
        lines.append('|----------|-------|')
        lines.append('| Protein | {} |'.format(protein_name or 'Unknown'))
        lines.append('| Enzyme class | {} |'.format(enzyme_class or 'Unknown'))
        lines.append('| Ligand | {} |'.format(ligand or 'Unknown'))
        lines.append('| Active site residues | {} |'.format(
            ', '.join(str(r) for r in sorted(binding_site))))
        lines.append('| Conserved positions | {} |'.format(
            ', '.join(str(r) for r in sorted(conserved))))
        lines.append('| KG brain online | {} |'.format(
            'Yes' if self.brain.is_online else 'No (offline mode)'))
        lines.append('| Total mutations screened | {} |'.format(p1.get('n_mutations', 0)))
        lines.append('| KG-enriched | {} |'.format(p1.get('n_kg_enriched', 0)))
        lines.append('')

        # Phase 2: Track A/B
        lines.append('## Phase 2: Structural Decomposition')
        lines.append('')
        lines.append('### Track A - Core (within 8A of active site)')
        lines.append('')
        lines.append('| Rank | Mutation | ZZ Score | Chem | PLM | Weights | Role |')
        lines.append('|:----:|----------|:--------:|:----:|:---:|:-------:|------|')
        for i, m in enumerate(p2.get('track_a', [])[:10], 1):
            lines.append('| {} | {} | {:.3f} | {:+.2f} | {:.3f} | C{:.0f}/P{:.0f} | {} |'.format(
                i, m['mutation'], m['zz_score'], m['chemical_score'],
                m.get('consensus_score', m.get('delta_ll', 0)),
                m['w_chem'] * 100, m['w_plm'] * 100, m['role']))
        lines.append('')

        lines.append('### Track B - Shell (channels, surface, distal)')
        lines.append('')
        lines.append('| Rank | Mutation | ZZ Score | Chem | PLM | Weights | Role |')
        lines.append('|:----:|----------|:--------:|:----:|:---:|:-------:|------|')
        for i, m in enumerate(p2.get('track_b', [])[:15], 1):
            lines.append('| {} | {} | {:.3f} | {:+.2f} | {:.3f} | C{:.0f}/P{:.0f} | {} |'.format(
                i, m['mutation'], m['zz_score'], m['chemical_score'],
                m.get('consensus_score', m.get('delta_ll', 0)),
                m['w_chem'] * 100, m['w_plm'] * 100, m['role']))
        lines.append('')

        # Phase 3: Combinations
        lines.append('## Phase 3: Multi-Mutation Combinations')
        lines.append('')
        lines.append('| Rank | Mutations | Strategy | Tracks | Adj Score | Epistasis |')
        lines.append('|:----:|-----------|----------|:------:|:---------:|:---------:|')
        for i, c in enumerate(combos[:15], 1):
            lines.append('| {} | {} | {} | {} | {:.3f} | {} |'.format(
                i, ' + '.join(c['mutations']), c['strategy'],
                '+'.join(c['tracks']), c['zz_adjusted'],
                c['epistasis']['risk_level']))
        lines.append('')

        # Phase 4: Validation
        lines.append('## Phase 4: Experimental Validation Plan')
        lines.append('')
        lines.append('### Tier 1: Singles')
        for m in p4.get('tier1_singles', []):
            lines.append('- **{}**'.format(m))
        lines.append('')
        lines.append('### Tier 2: Doubles')
        for c in p4.get('tier2_doubles', []):
            lines.append('- **{}**'.format(c))
        lines.append('')
        lines.append('### Tier 3: Triples')
        for c in p4.get('tier3_triples', []):
            lines.append('- **{}**'.format(c))
        lines.append('')
        lines.append('### MD Simulation Candidates')
        for m in p4.get('md_candidates', []):
            lines.append('- {} (active-site region, structural validation needed)'.format(m))
        lines.append('')

        # Chemical reasoning highlights
        lines.append('## Chemical Reasoning Highlights')
        lines.append('')
        lines.append('| Mutation | Chem | Key Reasoning |')
        lines.append('|----------|:----:|----|')
        for m in scored[:20]:
            chem = m.get('chemical_score', 0)
            if abs(chem) < 0.01:
                continue
            reasons = m.get('reasoning', []) + m.get('warnings', []) + m.get('opportunities', [])
            top_reason = reasons[0] if reasons else 'No specific chemical signal'
            # Truncate long reasons
            if len(top_reason) > 80:
                top_reason = top_reason[:77] + '...'
            lines.append('| {} | {:+.2f} | {} |'.format(
                m['mutation'], chem, top_reason))
        lines.append('')

        # Key insight
        lines.append('## Key Design Principle')
        lines.append('')
        lines.append('**Conserved active-site positions**: Chemical intuition weighted 80%.')
        lines.append('PLMs predict evolutionary consensus, not maximal catalytic activity.')
        lines.append('For optimized sequences, PLM scores at conserved sites often push')
        lines.append('back toward wild-type — the wrong direction for engineering.')
        lines.append('')
        lines.append('**Non-conserved distal positions**: PLM weighted 80%.')
        lines.append('Evolutionary signal is reliable for stability/expression effects.')
        lines.append('')

        report_text = '\n'.join(lines)
        report_path = os.path.join(output_dir, 'zhuangzhou_full_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        return report_text

    def _print_summary(self, scored, combos):
        """Print console summary."""
        print('\nTop 5 Singles (Zhuangzhou Score):')
        for m in scored[:5]:
            flag = ' [CONSERVED]' if m.get('conservation_flag', '').startswith('CONSERVED') else ''
            print('  {} ZZ={:.3f} (C{:.0f}/P{:.0f}) Track {} {}{}'.format(
                m['mutation'], m['zz_score'],
                m['w_chem'] * 100, m['w_plm'] * 100,
                m['track'], m['role'], flag))

        print('\nTop 5 Combinations:')
        for c in combos[:5]:
            print('  {} [{}] adj={:.3f} epistasis={}'.format(
                ' + '.join(c['mutations']), c['strategy'],
                c['zz_adjusted'], c['epistasis']['risk_level']))

    def _clean_for_json(self, obj):
        """Clean objects for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()
                    if k not in ('residue_coords', 'residue_names', 'structure')}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj
