"""
ProteinResolver — resolves protein identity from multiple sources.

Priority order:
1. PDB DBREF records (fastest, has UniProt ID)
2. PDB HEADER/COMPND/SOURCE records
3. UniProt REST API by ID
4. UniProt search by protein name
5. RCSB PDB API by PDB ID

All methods return a dict with standardized keys, never raise.
"""
import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional, Dict


_HEADERS = {'Accept': 'application/json', 'User-Agent': 'Zhuangzhou-HuiZi/1.0'}
_TIMEOUT = 10

_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
    # Modified residues
    'MSE': 'M', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'CYX': 'C', 'CYM': 'C',
    'GLH': 'E', 'ASH': 'D', 'LYN': 'K',
}


def _http_get(url: str, headers: dict = None, timeout: int = _TIMEOUT) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers=headers or _HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception:
        return None


def _http_get_text(url: str, timeout: int = _TIMEOUT) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Zhuangzhou-HuiZi/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8')
    except Exception:
        return None


class ProteinResolver:
    """Resolves protein identity from PDB files, UniProt, and RCSB PDB."""

    def from_pdb_file(self, pdb_path: str) -> dict:
        """
        Full PDB file parsing: DBREF + HEADER/COMPND/SOURCE.
        Returns merged result; auto-queries UniProt if DBREF has accession.
        """
        result = {}
        if not pdb_path or not os.path.exists(pdb_path):
            return result

        # DBREF has the most reliable cross-references
        dbref = self.from_pdb_dbref(pdb_path)
        header = self.from_pdb_header(pdb_path)

        # Merge: DBREF takes priority for IDs, HEADER for names
        result.update(header)
        for k, v in dbref.items():
            if v and not result.get(k):
                result[k] = v

        # Extract FASTA from SEQRES
        fasta = self.extract_fasta_from_pdb(pdb_path)
        if fasta:
            result['fasta_sequence'] = fasta

        # If we have UniProt ID from DBREF, fetch full info
        if result.get('uniprot_id'):
            uniprot_info = self.from_uniprot_id(result['uniprot_id'])
            for k, v in uniprot_info.items():
                if v and not result.get(k):
                    result[k] = v

        result['pdb_path'] = pdb_path
        return result

    def from_pdb_dbref(self, pdb_path: str) -> dict:
        """Parse DBREF records for database cross-references."""
        result = {}
        if not pdb_path or not os.path.exists(pdb_path):
            return result
        try:
            with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('DBREF'):
                        parts = line.split()
                        if len(parts) >= 7 and parts[5] in ('UNP', 'UNIPROT'):
                            result['uniprot_id'] = parts[6]
                        if len(parts) >= 2 and len(parts[1]) == 4:
                            result['pdb_id'] = parts[1]
                    elif line.startswith('ATOM'):
                        break
        except Exception:
            pass
        return result

    def from_pdb_header(self, pdb_path: str) -> dict:
        """Parse HEADER/COMPND/SOURCE/REMARK records."""
        result = {}
        if not pdb_path or not os.path.exists(pdb_path):
            return result

        try:
            compnd_lines, source_lines, remark_lines = [], [], []
            header_line = ""

            with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    rec = line[:6].strip()
                    if rec == 'HEADER':
                        header_line = line[10:50].strip()
                    elif rec == 'COMPND':
                        compnd_lines.append(line[10:].strip())
                    elif rec == 'SOURCE':
                        source_lines.append(line[10:].strip())
                    elif rec == 'REMARK':
                        remark_lines.append(line[11:].strip())
                    elif rec == 'ATOM':
                        break

            # HEADER: classification + PDB ID
            if header_line:
                parts = header_line.rsplit(None, 1)
                if len(parts) == 2 and len(parts[-1]) == 4:
                    result['pdb_id'] = parts[-1]

            # COMPND: protein name
            compnd_text = ' '.join(compnd_lines)
            mol_name = re.search(r'MOLECULE:\s*([^;]+)', compnd_text)
            if mol_name:
                name = mol_name.group(1).strip().rstrip(';')
                if name and name.upper() not in ('NULL', 'CHAIN', ''):
                    result['protein_name'] = name

            # SOURCE: organism, gene
            source_text = ' '.join(source_lines)
            org = re.search(r'ORGANISM_SCIENTIFIC:\s*([^;]+)', source_text)
            if org:
                result['organism'] = org.group(1).strip().rstrip(';')
            gene = re.search(r'GENE:\s*([^;]+)', source_text)
            if gene:
                result['gene_name'] = gene.group(1).strip().rstrip(';').split(',')[0]

            # REMARK: UniProt cross-reference
            for rline in remark_lines:
                up = re.search(r'UniProt[:\s]+([A-Z][A-Z0-9]{5})', rline, re.I)
                if up:
                    result['uniprot_id'] = up.group(1)
                    break

        except Exception:
            pass

        return result

    def extract_fasta_from_pdb(self, pdb_path: str) -> Optional[str]:
        """Extract amino acid sequence from SEQRES (or ATOM CA) records."""
        if not pdb_path or not os.path.exists(pdb_path):
            return None
        try:
            seqres: Dict[str, list] = {}
            with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('SEQRES'):
                        chain = line[11]
                        residues = line[19:70].split()
                        seqres.setdefault(chain, []).extend(
                            _3TO1.get(r.upper(), 'X') for r in residues
                        )

            if seqres:
                chain = sorted(seqres.keys())[0]
                return ''.join(seqres[chain])

            # Fallback: CA atoms
            seen: Dict[tuple, str] = {}
            with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        chain = line[21]
                        try:
                            resnum = int(line[22:26].strip())
                        except ValueError:
                            continue
                        resname = line[17:20].strip()
                        seen.setdefault((chain, resnum), _3TO1.get(resname, 'X'))

            if seen:
                chain = sorted(set(k[0] for k in seen))[0]
                return ''.join(v for k, v in sorted(seen.items()) if k[0] == chain)
        except Exception:
            pass
        return None

    def from_uniprot_id(self, uniprot_id: str) -> dict:
        """Fetch protein info from UniProt by accession."""
        if not uniprot_id:
            return {}
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        data = _http_get(url)
        return self._parse_uniprot_entry(data) if data else {}

    def from_name(self, protein_name: str, organism: str = None) -> dict:
        """Search UniProt by protein name."""
        if not protein_name:
            return {}
        query = f'protein_name:"{protein_name}" AND reviewed:true'
        if organism:
            query += f' AND organism_name:"{organism}"'
        url = (
            f"https://rest.uniprot.org/uniprotkb/search?"
            f"query={urllib.parse.quote(query)}&format=json&size=1"
            f"&fields=accession,protein_name,gene_names,organism_name,"
            f"cc_function,ec,ft_act_site,sequence"
        )
        data = _http_get(url)
        results = (data or {}).get('results', [])
        if not results:
            # Retry without reviewed filter
            url2 = url.replace(' AND reviewed:true', '')
            data2 = _http_get(url2)
            results = (data2 or {}).get('results', [])
        return self._parse_uniprot_entry(results[0]) if results else {}

    def fetch_fasta_from_uniprot(self, uniprot_id: str) -> Optional[str]:
        """Fetch raw FASTA sequence from UniProt."""
        if not uniprot_id:
            return None
        text = _http_get_text(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
        if not text:
            return None
        lines = text.strip().split('\n')
        return ''.join(l for l in lines if not l.startswith('>')) or None

    def _parse_uniprot_entry(self, entry: dict) -> dict:
        if not entry:
            return {}
        result = {}
        try:
            result['uniprot_id'] = entry.get('primaryAccession', '')
            pn = entry.get('proteinDescription', {})
            rec = pn.get('recommendedName', {})
            result['protein_name'] = rec.get('fullName', {}).get('value', '')
            genes = entry.get('genes', [])
            if genes:
                result['gene_name'] = genes[0].get('geneName', {}).get('value', '')
            result['organism'] = entry.get('organism', {}).get('scientificName', '')
            ec_list = [e.get('value', '') for e in rec.get('ecNumbers', [])]
            result['ec_number'] = ', '.join(ec_list) if ec_list else None
            for c in entry.get('comments', []):
                if c.get('commentType') == 'FUNCTION':
                    texts = c.get('texts', [])
                    if texts:
                        result['function'] = texts[0].get('value', '')
                        break
            features = entry.get('features', [])
            result['active_site_residues'] = [
                str(f.get('location', {}).get('start', {}).get('value', ''))
                for f in features
                if f.get('type') in ('Active site', 'Binding site')
                and f.get('location', {}).get('start', {}).get('value')
            ]
            seq_obj = entry.get('sequence', {})
            result['fasta_sequence'] = seq_obj.get('value', None)
        except Exception:
            pass
        return result
