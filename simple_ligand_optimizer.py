#!/usr/bin/env python3

import os
import re
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BranchRecord:
    """简化的BRANCH记录"""
    id: int
    start_line: int
    end_line: int
    axis_atoms: Tuple[int, int]
    depth: int
    parent_id: Optional[int]
    atom_types: Tuple[str, str]
    is_aromatic: bool = False
    score: float = 0.0


class SimpleLigandOptimizer:
    """
    简化版配体优化器
    
    核心策略：
    1. 解析BRANCH结构（不依赖复杂的atom tracking）
    2. 基于化学规则和简单启发式打分
    3. 直接操作文本，确保输出正确
    """
    
    # 芳香原子类型（PDBQT格式）
    AROMATIC_TYPES = {'A', 'NA', 'NS', 'SA'}
    
    # 极性/氢键原子
    POLAR_TYPES = {'N', 'NA', 'NS', 'O', 'OA', 'OS', 'S', 'SA'}
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.branches: List[BranchRecord] = []
        
    def optimize_to_target(self, 
                          input_pdbqt: str,
                          target_torsdof: int = 10,
                          output_pdbqt: Optional[str] = None) -> Tuple[str, Dict]:
        """
        优化配体到目标TORSDOF
        
        Returns:
            (output_file, report_dict)
        """
        
        if output_pdbqt is None:
            base = Path(input_pdbqt).stem
            output_pdbqt = str(Path(input_pdbqt).parent / f"{base}_optimized.pdbqt")
        
        # Step 1: 解析文件
        with open(input_pdbqt, 'r') as f:
            lines = f.readlines()
        
        original_torsdof = self._count_branches(lines)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Simple Ligand Optimizer v3.0")
            print(f"{'='*70}")
            print(f"Input: {Path(input_pdbqt).name}")
            print(f"Original TORSDOF: {original_torsdof}")
            print(f"Target TORSDOF: {target_torsdof}")
            print(f"{'='*70}\n")
        
        # 如果已经满足目标
        if original_torsdof <= target_torsdof:
            if self.verbose:
                print(f"✓ Already optimal (TORSDOF ≤ {target_torsdof})")
            
            # 直接复制文件
            with open(output_pdbqt, 'w') as f:
                f.writelines(lines)
            
            return output_pdbqt, {
                'original_torsdof': original_torsdof,
                'final_torsdof': original_torsdof,
                'frozen': 0,
                'kept': original_torsdof
            }
        
        # Step 2: 解析BRANCH结构
        self.branches = self._parse_branches(lines)
        
        if self.verbose:
            print(f"[1/3] Parsed {len(self.branches)} branches")
        
        # Step 3: 打分和选择
        self._score_branches(lines)
        
        # 需要移除的数量
        n_to_remove = original_torsdof - target_torsdof
        
        # 按分数排序（低分优先移除）
        sorted_branches = sorted(self.branches, key=lambda x: x.score)
        
        # 选择要移除的分支（分数最低的n个）
        branches_to_remove = sorted_branches[:n_to_remove]
        branches_to_keep = sorted_branches[n_to_remove:]
        
        remove_ids = {b.id for b in branches_to_remove}
        
        if self.verbose:
            print(f"[2/3] Selecting branches to freeze:")
            print(f"      Remove: {len(branches_to_remove)} (lowest scores)")
            print(f"      Keep: {len(branches_to_keep)} (highest scores)")
            print(f"      Score range (keep): {branches_to_keep[0].score:.1f} - {branches_to_keep[-1].score:.1f}")
        
        # Step 4: 生成优化后的文件
        new_lines = self._remove_branches(lines, remove_ids)
        
        # Step 5: 验证并写入
        final_torsdof = self._count_branches(new_lines)
        
        # 更新TORSDOF行
        new_lines = self._update_torsdof_line(new_lines, final_torsdof)
        
        with open(output_pdbqt, 'w') as f:
            f.writelines(new_lines)
        
        if self.verbose:
            print(f"[3/3] Generated optimized PDBQT:")
            print(f"      Final TORSDOF: {final_torsdof}")
            print(f"      Reduction: {original_torsdof - final_torsdof}")
            print(f"      Saved to: {Path(output_pdbqt).name}")
            print(f"{'='*70}\n")
        
        # 验证输出
        if final_torsdof != target_torsdof:
            print(f"⚠️  Warning: Final TORSDOF ({final_torsdof}) ≠ Target ({target_torsdof})")
        
        return output_pdbqt, {
            'original_torsdof': original_torsdof,
            'final_torsdof': final_torsdof,
            'frozen': len(branches_to_remove),
            'kept': len(branches_to_keep)
        }
    
    def _count_branches(self, lines: List[str]) -> int:
        """统计BRANCH数量"""
        return sum(1 for l in lines if l.startswith('BRANCH'))
    
    def _parse_branches(self, lines: List[str]) -> List[BranchRecord]:
        """解析BRANCH结构（简化版，只记录基本信息）"""
        
        branches = []
        branch_stack = []
        branch_id = 0
        
        for line_num, line in enumerate(lines):
            if line.startswith('BRANCH'):
                parts = line.split()
                if len(parts) >= 3:
                    atom1 = int(parts[1])
                    atom2 = int(parts[2])
                    
                    # 获取原子类型
                    type1, type2 = self._get_atom_types(lines, atom1, atom2)
                    
                    # 判断是否芳香
                    is_aromatic = (type1 in self.AROMATIC_TYPES or 
                                 type2 in self.AROMATIC_TYPES)
                    
                    parent_id = branch_stack[-1]['id'] if branch_stack else None
                    
                    branch_stack.append({
                        'id': branch_id,
                        'start': line_num,
                        'atom1': atom1,
                        'atom2': atom2,
                        'type1': type1,
                        'type2': type2,
                        'is_aromatic': is_aromatic,
                        'depth': len(branch_stack),
                        'parent_id': parent_id
                    })
                    
                    branch_id += 1
            
            elif line.startswith('ENDBRANCH'):
                if branch_stack:
                    info = branch_stack.pop()
                    
                    branch = BranchRecord(
                        id=info['id'],
                        start_line=info['start'],
                        end_line=line_num,
                        axis_atoms=(info['atom1'], info['atom2']),
                        depth=info['depth'],
                        parent_id=info['parent_id'],
                        atom_types=(info['type1'], info['type2']),
                        is_aromatic=info['is_aromatic']
                    )
                    
                    branches.append(branch)
        
        return branches
    
    def _get_atom_types(self, lines: List[str], atom1: int, atom2: int) -> Tuple[str, str]:
        """从PDBQT获取原子类型"""
        
        type1, type2 = 'C', 'C'  # 默认
        
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')):
                parts = line.split()
                if len(parts) >= 12:
                    try:
                        serial = int(parts[1])
                        atype = parts[11]
                        
                        if serial == atom1:
                            type1 = atype
                        if serial == atom2:
                            type2 = atype
                    except:
                        pass
        
        return type1, type2
    
    def _score_branches(self, lines: List[str]):
        """
        给BRANCH打分
        
        打分原则：
        - 分数越高，越重要，优先保留
        - 分数越低，优先移除
        """
        
        for branch in self.branches:
            score = 50.0  # 基础分
            
            type1, type2 = branch.atom_types
            
            # === 化学约束（降低分数，优先移除）===
            
            # 1. 芳香环内部键 - 严格禁止旋转
            if type1 in self.AROMATIC_TYPES and type2 in self.AROMATIC_TYPES:
                score -= 100
            
            # 2. 连接芳香原子 - 通常是重要的骨架
            elif type1 in self.AROMATIC_TYPES or type2 in self.AROMATIC_TYPES:
                score += 20  # 提高分数，保留
            
            # 3. 极性原子 - 可能参与氢键
            if type1 in self.POLAR_TYPES or type2 in self.POLAR_TYPES:
                score += 15
            
            # 4. 深度嵌套 - 末端旋转重要性低
            if branch.depth > 2:
                score -= 10
            elif branch.depth == 0:
                score += 10  # 顶层分支更重要
            
            # 5. 非极性碳链（C-C）- 旋转重要性相对较低
            if type1 == 'C' and type2 == 'C':
                score -= 5
            
            branch.score = score
    
    def _remove_branches(self, lines: List[str], remove_ids: Set[int]) -> List[str]:
        """
        移除指定的BRANCH（改进版：确保BRANCH/ENDBRANCH完全配对）
        
        策略：
        1. 跟踪每个BRANCH的ID
        2. 遇到要删除的BRANCH时，跳过整个块（包括所有嵌套内容）
        3. 使用栈确保BRANCH/ENDBRANCH完全配对删除
        """
        
        new_lines = []
        branch_stack = []  # 栈：[(branch_id, should_skip)]
        branch_counter = -1
        
        for line in lines:
            # 遇到BRANCH
            if line.startswith('BRANCH'):
                branch_counter += 1
                
                # 检查是否应该跳过这个分支
                should_skip = branch_counter in remove_ids
                
                # 如果父分支被跳过，子分支也跳过
                if branch_stack and branch_stack[-1][1]:
                    should_skip = True
                
                branch_stack.append((branch_counter, should_skip))
                
                # 如果不跳过，保留这行
                if not should_skip:
                    new_lines.append(line)
                
                continue
            
            # 遇到ENDBRANCH
            elif line.startswith('ENDBRANCH'):
                if branch_stack:
                    branch_id, should_skip = branch_stack.pop()
                    
                    # 只有不跳过的分支才保留ENDBRANCH
                    if not should_skip:
                        new_lines.append(line)
                else:
                    # 孤立的ENDBRANCH（不应该发生，但保留以防万一）
                    new_lines.append(line)
                
                continue
            
            # TORSDOF行跳过（稍后重新生成）
            if line.startswith('TORSDOF'):
                continue
            
            # 其他行：只有当前不在被跳过的分支内时才保留
            if not branch_stack or not branch_stack[-1][1]:
                new_lines.append(line)
        
        return new_lines
    
    def _update_torsdof_line(self, lines: List[str], torsdof: int) -> List[str]:
        """更新或添加TORSDOF行"""
        
        # 在文件末尾附近插入TORSDOF
        # 通常在最后一个ENDBRANCH之后
        
        # 找到最后一个ENDBRANCH的位置
        last_endbranch = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('ENDBRANCH'):
                last_endbranch = i
                break
        
        if last_endbranch >= 0:
            # 在ENDBRANCH后插入
            lines.insert(last_endbranch + 1, f"TORSDOF {torsdof}\n")
        else:
            # 没有ENDBRANCH，添加到末尾
            lines.append(f"TORSDOF {torsdof}\n")
        
        return lines


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple Ligand Optimizer - Reduce ligand flexibility"
    )
    parser.add_argument("input", help="Input PDBQT file")
    parser.add_argument("-t", "--target", type=int, default=10,
                       help="Target TORSDOF (default: 10)")
    parser.add_argument("-o", "--output", help="Output PDBQT file")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Quiet mode")
    
    args = parser.parse_args()
    
    optimizer = SimpleLigandOptimizer(verbose=not args.quiet)
    
    output_file, report = optimizer.optimize_to_target(
        args.input,
        target_torsdof=args.target,
        output_pdbqt=args.output
    )
    
    if not args.quiet:
        print(f"\n✓ Optimization complete!")
        print(f"  Input: {args.input}")
        print(f"  Output: {output_file}")
        print(f"  TORSDOF: {report['original_torsdof']} → {report['final_torsdof']}")


if __name__ == "__main__":
    main()
