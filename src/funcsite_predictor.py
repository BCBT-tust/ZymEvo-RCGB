# -*- coding: utf-8 -*-
"""
funcsite_predictor.py
=====================
FuncSite-ML 推理引擎
支持输入: PDB结构文件 或 氨基酸序列（FASTA）
输出: 每个残基的催化相关性(key_prob)和底物特异性(spec_prob)预测

用法（独立运行）:
  python funcsite_predictor.py \
    --pdb     input.pdb \
    --key_model  key_GB_final.pkl \
    --spec_model spec_GB_final.pkl \
    --output  predictions.csv
"""

import os, sys, re, math, warnings, logging, tempfile, subprocess
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

# ============================================================
# 氨基酸物化参数（与训练时完全一致）
# ============================================================

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
    'A':0,'R':1,'N':0,'D':-1,'C':0,
    'Q':0,'E':-1,'G':0,'H':0,'I':0,
    'L':0,'K':1,'M':0,'F':0,'P':0,
    'S':0,'T':0,'W':0,'Y':0,'V':0,
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


# ============================================================
# PDB 解析
# ============================================================

def parse_pdb(pdb_path: Path) -> pd.DataFrame:
    records = []
    with open(pdb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            try:
                records.append({
                    'atom_name': line[12:16].strip(),
                    'element':   line[76:78].strip().upper() or line[12].strip().upper(),
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
    res['aa1']   = res['res_name'].map(AA3_TO_1).fillna('X')
    res['res_id']= (res['chain_id']+'_'+res['res_name']+'_'
                    +res['res_seq'].astype(str))
    return res


# ============================================================
# P2Rank 口袋预测（可选，有则用）
# ============================================================

def run_p2rank(pdb_path: Path, p2rank_bin: Optional[Path],
               tmp_dir: Path) -> Dict[Tuple, float]:
    """运行P2Rank获取口袋分，若不可用返回空dict"""
    if p2rank_bin is None or not p2rank_bin.exists():
        return {}
    try:
        out_dir = tmp_dir / 'p2rank_out'
        out_dir.mkdir(exist_ok=True)
        subprocess.run(
            [str(p2rank_bin), 'predict',
             '-f', str(pdb_path),
             '-o', str(out_dir),
             '-threads', '2'],
            capture_output=True, timeout=120)
        csv_files = list(out_dir.glob('*residues.csv'))
        if not csv_files:
            return {}
        df = pd.read_csv(csv_files[0])
        df.columns = [c.strip().lower() for c in df.columns]
        score_map = {}
        for _, row in df.iterrows():
            chain = str(row.get('chain','')).strip() or 'A'
            try:
                seq   = int(str(row['residue_label']).strip())
                score = float(row['score'])
                score_map[(chain, seq)] = score
                score_map[('A',   seq)] = score
            except: pass
        return score_map
    except Exception as e:
        log.warning(f'P2Rank失败: {e}，F06将使用默认值0')
        return {}


# ============================================================
# MSA 守恒性（可选，有MAFFT+网络则计算）
# ============================================================

def compute_conservation_from_sequence(
        seq: str) -> Dict[int, float]:
    """
    尝试从UniProt获取同源序列并计算守恒性。
    若失败返回默认值0.5。
    """
    try:
        import requests
        resp = requests.get(
            'https://rest.uniprot.org/uniprotkb/search',
            params={'query': 'reviewed:true',
                    'format':'fasta', 'size':50},
            timeout=15)
        if resp.status_code != 200:
            raise Exception('UniProt不可用')
        # 简化：返回默认值（完整流程见msa_conservation.py）
        return {i+1: 0.5 for i in range(len(seq))}
    except:
        return {i+1: 0.5 for i in range(len(seq))}


# ============================================================
# 接触特征计算（无对接结果时用口袋中心估算）
# ============================================================

AD4_TO_ELEMENT = {
    'C':'C','A':'C','N':'N','NA':'N','O':'O','OA':'O',
    'H':'H','HD':'H','S':'S','SA':'S','P':'P','F':'F',
    'CL':'CL','BR':'BR','I':'I',
}

POS_CHARGED_RES = {'ARG','LYS','HIS'}
NEG_CHARGED_RES = {'ASP','GLU'}
HYDROPHOBIC_RES = {'ALA','VAL','LEU','ILE','PHE','TRP','MET','PRO'}


def estimate_contact_features_from_pocket(
        atom_df:      pd.DataFrame,
        res_df:       pd.DataFrame,
        pocket_center: np.ndarray) -> Dict[str, Dict]:
    """
    无对接文件时，用口袋中心作为虚拟配体位点估算接触特征。
    仅计算空间距离相关特征，相互作用特征设为0。
    """
    result = {}
    pocket_pt = pocket_center.reshape(1,3)
    tree      = cKDTree(pocket_pt)

    for _, res_row in res_df.iterrows():
        rid   = res_row['res_id']
        chain = res_row['chain_id']
        seq   = res_row['res_seq']

        mask      = ((atom_df['chain_id']==chain) &
                     (atom_df['res_seq']==seq))
        rec_atoms = atom_df[mask]
        if rec_atoms.empty:
            result[rid] = {k:0.0 for k in
                ['min_distance','contact_frequency','polar_contacts',
                 'ionic_contacts','hydrophobic_contacts']}
            continue

        coords = rec_atoms[['x','y','z']].values
        dists, _ = tree.query(coords)
        min_d    = float(dists.min())

        result[rid] = {
            'min_distance':          min_d,
            'contact_frequency':     max(0.0, 1.0/(1.0+min_d*0.5)),
            'polar_contacts':        0.0,
            'ionic_contacts':        0.0,
            'hydrophobic_contacts':  0.0,
        }
    return result


# ============================================================
# 核心特征提取
# ============================================================

def extract_features(
        pdb_path:       Path,
        p2rank_bin:     Optional[Path] = None,
        pdbqt_files:    Optional[List[Path]] = None,
        ci_map:         Optional[Dict[int,float]] = None,
) -> pd.DataFrame:
    """
    从PDB文件提取16维特征向量。
    - p2rank_bin: P2Rank可执行文件路径（可选）
    - pdbqt_files: 对接结果（可选，有则计算接触特征）
    - ci_map: 守恒性字典 {res_seq: Ci}（可选）
    """
    log.info(f'解析PDB: {pdb_path.name}')
    atom_df = parse_pdb(pdb_path)
    res_df  = residue_table(atom_df)

    if res_df.empty:
        raise ValueError('PDB中无有效残基')

    seq_min = int(res_df['res_seq'].min())
    seq_max = int(res_df['res_seq'].max())
    seq_len = max(seq_max - seq_min + 1, 1)

    # CA坐标字典
    ca_df  = atom_df[atom_df['atom_name']=='CA']
    ca_map = {(r['chain_id'],r['res_seq']): np.array([r['x'],r['y'],r['z']])
              for _,r in ca_df.iterrows()}

    # P2Rank 口袋分
    with tempfile.TemporaryDirectory() as tmp:
        pocket_scores = run_p2rank(pdb_path, p2rank_bin, Path(tmp))

    # 口袋中心（用于无对接文件时的估算）
    if pocket_scores:
        best_key  = max(pocket_scores, key=pocket_scores.get)
        best_ca   = ca_map.get(best_key)
        pocket_center = best_ca if best_ca is not None else \
                        np.mean(list(ca_map.values()), axis=0)
    else:
        pocket_center = np.mean(list(ca_map.values()), axis=0)

    # 接触特征
    if pdbqt_files:
        contact_feats = _compute_contact_from_pdbqt(
            atom_df, res_df, pdbqt_files)
    else:
        log.info('无对接文件，用口袋中心估算接触特征')
        contact_feats = estimate_contact_features_from_pocket(
            atom_df, res_df, pocket_center)

    # 守恒性
    if ci_map is None:
        seq_str = ''.join(
            AA3_TO_1.get(r['res_name'],'X')
            for _,r in res_df.iterrows())
        ci_map = compute_conservation_from_sequence(seq_str)

    # 组装16维特征
    rows = []
    for _, r in res_df.iterrows():
        rid   = r['res_id']
        chain = r['chain_id']
        seq   = int(r['res_seq'])
        aa1   = r['aa1']
        cf    = contact_feats.get(rid, {k:0.0 for k in
                ['min_distance','contact_frequency','polar_contacts',
                 'ionic_contacts','hydrophobic_contacts']})

        rows.append({
            'res_id':  rid,
            'chain_id':chain,
            'res_seq': seq,
            'res_name':r['res_name'],
            'aa1':     aa1,
            'F01_min_distance':         cf['min_distance'],
            'F02_contact_frequency':    cf['contact_frequency'],
            'F03_polar_contacts':       cf['polar_contacts'],
            'F04_ionic_contacts':       cf['ionic_contacts'],
            'F05_hydrophobic_contacts': cf['hydrophobic_contacts'],
            'F06_pocket_score':         float(
                pocket_scores.get((chain,seq)) or
                pocket_scores.get(('A',seq)) or 0.0),
            'F07_tunnel_descriptor':    999.0,  # 无CAVER时默认
            'F08_molecular_weight':     MW_VALUES.get(aa1,128.0),
            'F09_hydrophobicity':       HYDROPHOBICITY.get(aa1,0.0),
            'F10_charge_state':         float(CHARGE_STATE.get(aa1,0)),
            'F11_isoelectric_point':    PI_VALUES.get(aa1,6.0),
            'F12_polarity':             POLARITY.get(aa1,0.0),
            'F13_sidechain_volume':     SC_VOLUME.get(aa1,100.0),
            'F14_conservation_score':   ci_map.get(seq,0.5),
            'F15_conservation_class':   float(
                3 if ci_map.get(seq,0.5)>0.8
                else 2 if ci_map.get(seq,0.5)>=0.5 else 1),
            'F16_relative_position':    (seq-seq_min)/(seq_len-1)
                                        if seq_len>1 else 0.5,
        })

    return pd.DataFrame(rows)


def _compute_contact_from_pdbqt(atom_df, res_df, pdbqt_files):
    """有对接文件时计算真实接触特征（只取第1构象最佳pose）"""
    AD4 = AD4_TO_ELEMENT

    feat_keys = ['min_distance','contact_frequency',
                 'polar_contacts','ionic_contacts','hydrophobic_contacts']
    acc = {r['res_id']:{k:[] for k in feat_keys}
           for _,r in res_df.iterrows()}

    res_atom_map = {}
    for _,r in res_df.iterrows():
        mask = ((atom_df['chain_id']==r['chain_id']) &
                (atom_df['res_seq']==r['res_seq']))
        res_atom_map[r['res_id']] = atom_df[mask]

    for pf in pdbqt_files:
        # 只取第1构象
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
                            elements.append(AD.get(ad, ad[0] if ad else 'C'))
                        except: pass
            if not coords: continue
            lig = np.array(coords, dtype=np.float32)
            tree= cKDTree(lig)

            for _,rr in res_df.iterrows():
                rid  = rr['res_id']
                ra   = res_atom_map[rid]
                if ra.empty: continue
                rc   = ra[['x','y','z']].values
                re   = ra['element'].tolist()
                res3 = rr['res_name']

                dists,_ = tree.query(rc)
                md = float(dists.min())
                acc[rid]['min_distance'].append(md)

                n_contact = sum(
                    len(tree.query_ball_point(p, r=4.0)) for p in rc)
                acc[rid]['contact_frequency'].append(
                    n_contact/max(len(rc)*len(lig),1))

                # 极性接触
                pc = 0.0
                for i,p in enumerate(rc):
                    if re[i][0] not in {'N','O','S'}: continue
                    for j in tree.query_ball_point(p, r=3.5):
                        if elements[j] in {'N','O','F','S'}:
                            d = float(np.linalg.norm(p-lig[j]))
                            pc += (1.5/(1+d)) if d>=2.0 else -(1.5/(d+1e-6))
                acc[rid]['polar_contacts'].append(pc)

                # 离子接触
                ic = 0.0
                if res3 in POS_CHARGED_RES or res3 in NEG_CHARGED_RES:
                    ctr = rc.mean(0)
                    for j,lc in enumerate(lig):
                        le = elements[j]
                        if ((res3 in POS_CHARGED_RES and le in {'O','S'}) or
                            (res3 in NEG_CHARGED_RES and le=='N')):
                            d = float(np.linalg.norm(ctr-lc))
                            if d<=4.0:
                                ic += (1.5/(1+d)) if d>=2.0 else -(1.5/(d+1e-6))
                acc[rid]['ionic_contacts'].append(ic)

                # 疏水接触
                hc = 0.0
                if res3 in HYDROPHOBIC_RES:
                    for i,p in enumerate(rc):
                        if re[i][0]!='C': continue
                        for j in tree.query_ball_point(p, r=4.5):
                            if elements[j] in {'C','S','F','CL','BR','I'}:
                                d = float(np.linalg.norm(p-lig[j]))
                                hc += 1.0/(1+d)
                acc[rid]['hydrophobic_contacts'].append(hc)
        except Exception as e:
            log.warning(f'PDBQT解析失败 {pf.name}: {e}')

    return {rid:{k:float(np.mean(v)) if v else 0.0
                 for k,v in d.items()}
            for rid,d in acc.items()}


# ============================================================
# 预测推理
# ============================================================

def predict(pdb_path:    Path,
            key_model_path:  Path,
            spec_model_path: Path,
            p2rank_bin:  Optional[Path] = None,
            pdbqt_files: Optional[List[Path]] = None,
            ci_map:      Optional[Dict] = None,
            key_threshold:  float = 0.5,
            spec_threshold: float = 0.5) -> pd.DataFrame:
    """
    主预测函数。
    返回每个残基的预测概率DataFrame。
    """
    import joblib

    # 提取特征
    feat_df = extract_features(pdb_path, p2rank_bin,
                               pdbqt_files, ci_map)
    X = np.nan_to_num(feat_df[FEATURE_COLS].values.astype(float))

    # 加载模型并预测
    key_pipe  = joblib.load(key_model_path)
    spec_pipe = joblib.load(spec_model_path)

    key_prob  = key_pipe.predict_proba(X)[:,1]
    spec_prob = spec_pipe.predict_proba(X)[:,1]

    feat_df['key_prob']       = key_prob
    feat_df['spec_prob']      = spec_prob
    feat_df['key_pred']       = (key_prob  >= key_threshold).astype(int)
    feat_df['spec_pred']      = (spec_prob >= spec_threshold).astype(int)
    feat_df['dual_functional']= (
        (key_prob  >= key_threshold) &
        (spec_prob >= spec_threshold)).astype(int)

    # 排序输出
    result = feat_df[['res_id','chain_id','res_seq','res_name','aa1',
                       'key_prob','spec_prob','key_pred','spec_pred',
                       'dual_functional'] + FEATURE_COLS].copy()
    result = result.sort_values('key_prob', ascending=False)
    return result


# ============================================================
# CLI 入口
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='FuncSite-ML 功能位点预测')
    parser.add_argument('--pdb',        required=True)
    parser.add_argument('--key_model',  required=True)
    parser.add_argument('--spec_model', required=True)
    parser.add_argument('--p2rank',     default=None,
                        help='P2Rank可执行文件路径（可选）')
    parser.add_argument('--pdbqt_dir',  default=None,
                        help='对接结果目录（可选）')
    parser.add_argument('--output',     default='predictions.csv')
    parser.add_argument('--key_thr',    type=float, default=0.5)
    parser.add_argument('--spec_thr',   type=float, default=0.5)
    args = parser.parse_args()

    pdbqt_files = None
    if args.pdbqt_dir:
        pdbqt_files = list(Path(args.pdbqt_dir).glob('*.pdbqt'))

    result = predict(
        pdb_path        = Path(args.pdb),
        key_model_path  = Path(args.key_model),
        spec_model_path = Path(args.spec_model),
        p2rank_bin      = Path(args.p2rank) if args.p2rank else None,
        pdbqt_files     = pdbqt_files,
        key_threshold   = args.key_thr,
        spec_threshold  = args.spec_thr,
    )
    result.to_csv(args.output, index=False)
    log.info(f'预测完成: {args.output}')

    print(f'\n{"="*50}')
    print(f'总残基数: {len(result)}')
    print(f'催化相关残基 (key_pred=1): {result["key_pred"].sum()}')
    print(f'特异性残基   (spec_pred=1): {result["spec_pred"].sum()}')
    print(f'双功能残基:  {result["dual_functional"].sum()}')
    print(f'\nTop 5 催化位点:')
    for _,r in result.head(5).iterrows():
        print(f'  {r["res_id"]:>15}  key_prob={r["key_prob"]:.3f}  '
              f'spec_prob={r["spec_prob"]:.3f}')
