# -*- coding: utf-8 -*-
import os, re, math, warnings, logging, tempfile, subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'MSE':'M','HSD':'H','HSE':'H','HSP':'H',
}
HYDROPHOBICITY = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,
    'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,
    'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}
CHARGE_STATE = {
    'A':0,'R':1,'N':0,'D':-1,'C':0,'Q':0,'E':-1,
    'G':0,'H':0,'I':0,'L':0,'K':1,'M':0,'F':0,
    'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0,
}
PI_VALUES = {
    'A':6.00,'R':10.76,'N':5.41,'D':2.77,'C':5.07,
    'Q':5.65,'E':3.22,'G':5.97,'H':7.59,'I':6.02,
    'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.30,
    'S':5.68,'T':5.87,'W':5.89,'Y':5.66,'V':5.96,
}
POLARITY = {
    'A':0.00,'R':0.52,'N':0.36,'D':0.43,'C':0.17,
    'Q':0.35,'E':0.43,'G':0.00,'H':0.29,'I':0.00,
    'L':0.00,'K':0.49,'M':0.10,'F':0.04,'P':0.00,
    'S':0.24,'T':0.24,'W':0.13,'Y':0.20,'V':0.00,
}
SC_VOLUME = {
    'A':67.0,'R':148.0,'N':96.0,'D':91.0,'C':86.0,
    'Q':114.0,'E':109.0,'G':48.0,'H':118.0,'I':124.0,
    'L':124.0,'K':135.0,'M':124.0,'F':135.0,'P':90.0,
    'S':73.0,'T':93.0,'W':163.0,'Y':141.0,'V':105.0,
}
MW_VALUES = {
    'A':89.1,'R':174.2,'N':132.1,'D':133.1,'C':121.2,
    'Q':146.2,'E':147.1,'G':75.0,'H':155.2,'I':131.2,
    'L':131.2,'K':146.2,'M':149.2,'F':165.2,'P':115.1,
    'S':105.1,'T':119.1,'W':204.2,'Y':181.2,'V':117.1,
}

FEATURE_COLS = [
    'F01_min_distance','F02_contact_frequency',
    'F03_polar_contacts','F04_ionic_contacts','F05_hydrophobic_contacts',
    'F06_pocket_score','F07_tunnel_descriptor',
    'F08_molecular_weight','F09_hydrophobicity','F10_charge_state',
    'F11_isoelectric_point','F12_polarity','F13_sidechain_volume',
    'F14_conservation_score','F15_conservation_class',
    'F16_relative_position',
]

H_MAX = math.log2(20)
AA20  = list('ACDEFGHIKLMNPQRSTVWY')

POS_CHARGED_RES = {'ARG','LYS','HIS'}
NEG_CHARGED_RES = {'ASP','GLU'}
HYDROPHOBIC_RES = {'ALA','VAL','LEU','ILE','PHE','TRP','MET','PRO'}
AD4_TO_ELEMENT  = {
    'C':'C','A':'C','N':'N','NA':'N','O':'O','OA':'O',
    'H':'H','HD':'H','S':'S','SA':'S','P':'P','F':'F',
    'CL':'CL','BR':'BR','I':'I',
}

def parse_pdb(pdb_path: Path) -> pd.DataFrame:
    records = []
    with open(pdb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            try:
                elem = line[76:78].strip().upper()
                records.append({
                    'atom_name': line[12:16].strip(),
                    'element':   elem if elem else line[12:16].strip()[0].upper(),
                    'res_name':  line[17:20].strip(),
                    'chain_id':  line[21].strip() or 'A',
                    'res_seq':   int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(records)


def residue_table(atom_df: pd.DataFrame) -> pd.DataFrame:
    res = (atom_df[['res_name','chain_id','res_seq']]
           .drop_duplicates()
           .sort_values(['chain_id','res_seq'])
           .reset_index(drop=True))
    res['aa1'] = res['res_name'].map(AA3_TO_1).fillna('X')
    # 修复res_id格式：统一为 {chain}_{res3}_{seq}
    res['res_id'] = (res['chain_id'] + '_'
                     + res['res_name'] + '_'
                     + res['res_seq'].astype(str))
    return res

def run_p2rank(pdb_path: Path,
               p2rank_bin: Optional[Path],
               tmp_dir: Path) -> Dict[Tuple, float]:
    if p2rank_bin is None or not p2rank_bin.exists():
        return {}
    try:
        out = tmp_dir / 'p2rank_out'
        out.mkdir(exist_ok=True)
        subprocess.run(
            [str(p2rank_bin), 'predict',
             '-f', str(pdb_path), '-o', str(out), '-threads', '2'],
            capture_output=True, timeout=120)
        csvs = list(out.glob('*residues.csv'))
        if not csvs:
            return {}
        df = pd.read_csv(csvs[0])
        df.columns = [c.strip().lower() for c in df.columns]
        score_map = {}
        for _, row in df.iterrows():
            chain = str(row.get('chain', '')).strip() or 'A'
            try:
                seq   = int(str(row['residue_label']).strip())
                score = float(row['score'])
                score_map[(chain, seq)] = score
                score_map[('A',   seq)] = score
            except: pass
        return score_map
    except Exception as e:
        log.warning(f'P2Rank失败: {e}，F06使用默认值0')
        return {}

def _contact_features_from_pocket(atom_df, res_df,
                                   pocket_center: np.ndarray) -> Dict:
    """无对接文件时用口袋中心估算"""
    result = {}
    tree = cKDTree(pocket_center.reshape(1, 3))
    for _, r in res_df.iterrows():
        rid  = r['res_id']
        mask = ((atom_df['chain_id'] == r['chain_id']) &
                (atom_df['res_seq']  == r['res_seq']))
        atoms = atom_df[mask]
        if atoms.empty:
            result[rid] = {k: 0.0 for k in
                ['min_distance','contact_frequency','polar_contacts',
                 'ionic_contacts','hydrophobic_contacts']}
            continue
        coords = atoms[['x','y','z']].values
        dists, _ = tree.query(coords)
        md = float(dists.min())
        result[rid] = {
            'min_distance':         md,
            'contact_frequency':    max(0.0, 1.0/(1.0 + md*0.5)),
            'polar_contacts':       0.0,
            'ionic_contacts':       0.0,
            'hydrophobic_contacts': 0.0,
        }
    return result


def _contact_features_from_pdbqt(atom_df, res_df,
                                  pdbqt_files: List[Path]) -> Dict:
    feat_keys = ['min_distance','contact_frequency',
                 'polar_contacts','ionic_contacts','hydrophobic_contacts']
    acc = {r['res_id']: {k: [] for k in feat_keys}
           for _, r in res_df.iterrows()}

    res_atom_map = {}
    for _, r in res_df.iterrows():
        mask = ((atom_df['chain_id'] == r['chain_id']) &
                (atom_df['res_seq']  == r['res_seq']))
        res_atom_map[r['res_id']] = atom_df[mask]

    for pf in pdbqt_files:
        coords, elements = [], []
        in_m1 = False
        try:
            with open(pf, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('MODEL'):
                        if in_m1: break
                        in_m1 = True
                    elif line.startswith(('HETATM','ATOM')) and in_m1:
                        try:
                            x,y,z = (float(line[30:38]),
                                     float(line[38:46]),
                                     float(line[46:54]))
                            ad = line[77:79].strip().upper() if len(line)>77 else 'C'
                            coords.append([x,y,z])
                            elements.append(AD4_TO_ELEMENT.get(ad,
                                            ad[0] if ad else 'C'))
                        except: pass
            if not coords: continue
            lig  = np.array(coords, dtype=np.float32)
            tree = cKDTree(lig)

            for _, rr in res_df.iterrows():
                rid  = rr['res_id']
                ra   = res_atom_map[rid]
                if ra.empty: continue
                rc   = ra[['x','y','z']].values
                re_  = ra['element'].tolist()
                res3 = rr['res_name']

                dists, _ = tree.query(rc)
                md = float(dists.min())
                acc[rid]['min_distance'].append(md)

                n_ct = sum(len(tree.query_ball_point(p, r=4.0)) for p in rc)
                acc[rid]['contact_frequency'].append(
                    n_ct / max(len(rc)*len(lig), 1))

                pc = 0.0
                for i, p in enumerate(rc):
                    if re_[i][0] not in {'N','O','S'}: continue
                    for j in tree.query_ball_point(p, r=3.5):
                        if elements[j] in {'N','O','F','S'}:
                            d = float(np.linalg.norm(p - lig[j]))
                            pc += (1.5/(1+d)) if d >= 2.0 else -(1.5/(d+1e-6))
                acc[rid]['polar_contacts'].append(pc)

                ic = 0.0
                if res3 in POS_CHARGED_RES or res3 in NEG_CHARGED_RES:
                    ctr = rc.mean(0)
                    for j, lc in enumerate(lig):
                        le = elements[j]
                        if ((res3 in POS_CHARGED_RES and le in {'O','S'}) or
                            (res3 in NEG_CHARGED_RES and le == 'N')):
                            d = float(np.linalg.norm(ctr - lc))
                            if d <= 4.0:
                                ic += (1.5/(1+d)) if d >= 2.0 else -(1.5/(d+1e-6))
                acc[rid]['ionic_contacts'].append(ic)

                hc = 0.0
                if res3 in HYDROPHOBIC_RES:
                    for i, p in enumerate(rc):
                        if re_[i][0] != 'C': continue
                        for j in tree.query_ball_point(p, r=4.5):
                            if elements[j] in {'C','S','F','CL','BR','I'}:
                                d = float(np.linalg.norm(p - lig[j]))
                                hc += 1.0/(1+d)
                acc[rid]['hydrophobic_contacts'].append(hc)

        except Exception as e:
            log.warning(f'PDBQT解析失败 {pf.name}: {e}')

    return {rid: {k: float(np.mean(v)) if v else 0.0
                  for k, v in d.items()}
            for rid, d in acc.items()}

def get_conservation(seq: str) -> Dict[int, float]:
    try:
        import requests, subprocess as sp
        resp = requests.get(
            'https://rest.uniprot.org/uniprotkb/search',
            params={'query': f'reviewed:true', 'format': 'fasta', 'size': 30},
            timeout=10)
        if resp.status_code != 200:
            raise Exception()
        return {i+1: 0.5 for i in range(len(seq))}
    except:
        return {i+1: 0.5 for i in range(len(seq))}

def extract_features(pdb_path:    Path,
                     p2rank_bin:  Optional[Path] = None,
                     pdbqt_files: Optional[List[Path]] = None,
                     ci_map:      Optional[Dict] = None) -> pd.DataFrame:

    log.info(f'解析PDB: {pdb_path.name}')
    atom_df = parse_pdb(pdb_path)
    res_df  = residue_table(atom_df)
    if res_df.empty:
        raise ValueError('PDB中无有效残基')

    seq_min = int(res_df['res_seq'].min())
    seq_len = max(int(res_df['res_seq'].max()) - seq_min + 1, 1)

    ca_df  = atom_df[atom_df['atom_name'] == 'CA']
    ca_map = {(r['chain_id'], r['res_seq']):
              np.array([r['x'], r['y'], r['z']])
              for _, r in ca_df.iterrows()}

    with tempfile.TemporaryDirectory() as tmp:
        pocket_scores = run_p2rank(pdb_path, p2rank_bin, Path(tmp))

    if pocket_scores:
        best = max(pocket_scores, key=pocket_scores.get)
        pocket_center = ca_map.get(best,
                        np.mean(list(ca_map.values()), axis=0))
    else:
        pocket_center = np.mean(list(ca_map.values()), axis=0)

    if pdbqt_files:
        cf_map = _contact_features_from_pdbqt(atom_df, res_df, pdbqt_files)
    else:
        cf_map = _contact_features_from_pocket(atom_df, res_df, pocket_center)

    if ci_map is None:
        seq_str = ''.join(AA3_TO_1.get(r['res_name'], 'X')
                          for _, r in res_df.iterrows())
        ci_map = get_conservation(seq_str)

    rows = []
    for _, r in res_df.iterrows():
        rid   = r['res_id']
        chain = r['chain_id']
        seq   = int(r['res_seq'])
        aa1   = r['aa1']
        cf    = cf_map.get(rid, {k: 0.0 for k in
                ['min_distance','contact_frequency','polar_contacts',
                 'ionic_contacts','hydrophobic_contacts']})
        ci    = ci_map.get(seq, 0.5)

        rows.append({
            'chain_id':  chain,
            'res_seq':   seq,
            'res_name':  r['res_name'],
            'aa1':       aa1,
            'F01_min_distance':         cf['min_distance'],
            'F02_contact_frequency':    cf['contact_frequency'],
            'F03_polar_contacts':       cf['polar_contacts'],
            'F04_ionic_contacts':       cf['ionic_contacts'],
            'F05_hydrophobic_contacts': cf['hydrophobic_contacts'],
            'F06_pocket_score':         float(
                pocket_scores.get((chain, seq)) or
                pocket_scores.get(('A', seq)) or 0.0),
            'F07_tunnel_descriptor':    999.0,
            'F08_molecular_weight':     MW_VALUES.get(aa1, 128.0),
            'F09_hydrophobicity':       HYDROPHOBICITY.get(aa1, 0.0),
            'F10_charge_state':         float(CHARGE_STATE.get(aa1, 0)),
            'F11_isoelectric_point':    PI_VALUES.get(aa1, 6.0),
            'F12_polarity':             POLARITY.get(aa1, 0.0),
            'F13_sidechain_volume':     SC_VOLUME.get(aa1, 100.0),
            'F14_conservation_score':   ci,
            'F15_conservation_class':   float(3 if ci > 0.8
                                              else 2 if ci >= 0.5 else 1),
            'F16_relative_position':    (seq - seq_min) / (seq_len - 1)
                                        if seq_len > 1 else 0.5,
        })

    return pd.DataFrame(rows)

def predict(pdb_path:        Path,
            rf_key_model:    Path,
            rf_spec_model:   Path,
            gb_key_model:    Path,
            gb_spec_model:   Path,
            p2rank_bin:      Optional[Path] = None,
            pdbqt_files:     Optional[List[Path]] = None,
            ci_map:          Optional[Dict] = None,
            key_threshold:   float = 0.4,
            spec_threshold:  float = 0.4,
            rf_weight:       float = 0.5,
            gb_weight:       float = 0.5) -> pd.DataFrame:
    import joblib

    feat_df = extract_features(pdb_path, p2rank_bin, pdbqt_files, ci_map)
    X = np.nan_to_num(feat_df[FEATURE_COLS].values.astype(float))

    rf_key  = joblib.load(rf_key_model)
    rf_spec = joblib.load(rf_spec_model)
    gb_key  = joblib.load(gb_key_model)
    gb_spec = joblib.load(gb_spec_model)

    key_prob  = (rf_weight * rf_key.predict_proba(X)[:,1] +
                 gb_weight * gb_key.predict_proba(X)[:,1])
    spec_prob = (rf_weight * rf_spec.predict_proba(X)[:,1] +
                 gb_weight * gb_spec.predict_proba(X)[:,1])

    feat_df['key_prob']  = key_prob
    feat_df['spec_prob'] = spec_prob

    is_key  = key_prob  >= key_threshold
    is_spec = spec_prob >= spec_threshold

    def func_type(k, s):
        if k and s:  return 'Dual-functional'
        if k:        return 'Catalytic'
        if s:        return 'Specificity'
        return 'Background'

    feat_df['function_type'] = [func_type(k, s)
                                 for k, s in zip(is_key, is_spec)]

    feat_df['functional_score'] = np.maximum(key_prob, spec_prob)

    func_df = feat_df[feat_df['function_type'] != 'Background'].copy()
    func_df = func_df.sort_values('functional_score', ascending=False
                                  ).reset_index(drop=True)
    func_df.index += 1   # 排名从1开始

    func_df['position'] = (func_df['res_name'] + str('') +
                           func_df['res_seq'].astype(str))
    func_df['chain']    = func_df['chain_id']
    func_df['score']    = func_df['functional_score'].round(3)

    def confidence(score):
        if score >= 0.7: return 'High'
        if score >= 0.5: return 'Medium'
        return 'Low'

    func_df['confidence'] = func_df['score'].apply(confidence)

    output = func_df[[
        'position', 'chain', 'aa1',
        'function_type', 'score', 'confidence',
        'key_prob', 'spec_prob',
    ]].copy()
    output.columns = [
        'Residue', 'Chain', 'AA',
        'Function', 'Score', 'Confidence',
        'Catalytic_prob', 'Specificity_prob',
    ]
    output['Catalytic_prob']    = output['Catalytic_prob'].round(3)
    output['Specificity_prob']  = output['Specificity_prob'].round(3)

    full_df = feat_df[['chain_id','res_seq','res_name','aa1',
                       'key_prob','spec_prob','function_type',
                       'functional_score']].copy()
    full_df.columns = ['Chain','Position','Residue','AA',
                       'Catalytic_prob','Specificity_prob',
                       'Function','Score']
    full_df = full_df.sort_values('Score', ascending=False)
    full_df['Catalytic_prob']   = full_df['Catalytic_prob'].round(3)
    full_df['Specificity_prob'] = full_df['Specificity_prob'].round(3)
    full_df['Score']            = full_df['Score'].round(3)

    return output, full_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='FuncSite-ML v2 (RF+GB Ensemble)')
    parser.add_argument('--pdb',          required=True)
    parser.add_argument('--rf_key',       required=True)
    parser.add_argument('--rf_spec',      required=True)
    parser.add_argument('--gb_key',       required=True)
    parser.add_argument('--gb_spec',      required=True)
    parser.add_argument('--pdbqt_dir',    default=None)
    parser.add_argument('--p2rank',       default=None)
    parser.add_argument('--key_thr',      type=float, default=0.4)
    parser.add_argument('--spec_thr',     type=float, default=0.4)
    parser.add_argument('--output',       default='results')
    args = parser.parse_args()

    pdbqt = list(Path(args.pdbqt_dir).glob('*.pdbqt')) \
            if args.pdbqt_dir else None

    output, full = predict(
        pdb_path       = Path(args.pdb),
        rf_key_model   = Path(args.rf_key),
        rf_spec_model  = Path(args.rf_spec),
        gb_key_model   = Path(args.gb_key),
        gb_spec_model  = Path(args.gb_spec),
        p2rank_bin     = Path(args.p2rank) if args.p2rank else None,
        pdbqt_files    = pdbqt,
        key_threshold  = args.key_thr,
        spec_threshold = args.spec_thr,
    )

    print(f'\n{"="*55}')
    print(f' FuncSite-ML Results — {Path(args.pdb).stem}')
    print(f'{"="*55}')
    print(f' Functional residues identified: {len(output)}')
    for ft in ['Dual-functional','Catalytic','Specificity']:
        n = (output['Function'] == ft).sum()
        if n: print(f'   {ft}: {n}')
    print(f'{"="*55}\n')
    print(output.to_string())

    output.to_csv(f'{args.output}_summary.csv',  index_label='Rank')
    full.to_csv(  f'{args.output}_full.csv',     index=False)
    print(f'\nSaved: {args.output}_summary.csv  |  {args.output}_full.csv')
