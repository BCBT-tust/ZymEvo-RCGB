#!/usr/bin/env python3
"""
增强版柔性处理 - 添加详细调试信息
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional

class Config:
    MGLTOOLS_PATH = "/usr/local/autodocktools/bin/pythonsh"
    PREPARE_FLEXRECEPTOR = "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py"
    PYTHONPATH = "/usr/local/autodocktools/MGLToolsPckgs"
    TIMEOUT_SECONDS = 300


def check_residues_in_pdbqt(pdbqt_file: str, residues: str) -> Tuple[bool, str]:
    """检查指定的残基是否存在于 PDBQT 文件中"""
    print(f"\n🔍 检查残基存在性...")
    
    try:
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        # 获取文件中所有残基编号
        found_residues = {}  # {resid: chain}
        for line in lines:
            if line.startswith(('ATOM', 'HETATM')) and len(line) >= 27:
                try:
                    chain = line[21] if len(line) > 21 else ' '
                    res_num = line[22:27].strip()
                    if res_num:
                        key = f"{chain}:{res_num}" if chain != ' ' else res_num
                        found_residues[res_num] = chain
                except:
                    pass
        
        print(f"   文件中找到 {len(found_residues)} 个残基")
        print(f"   残基范围: {min(found_residues.keys())} - {max(found_residues.keys())}")
        
        # 检查指定的残基
        requested_residues = residues.split(':')
        print(f"   请求的残基: {requested_residues}")
        
        missing = []
        for res_id in requested_residues:
            if res_id not in found_residues:
                missing.append(res_id)
        
        if missing:
            msg = f"残基未找到: {missing}"
            print(f"   ❌ {msg}")
            print(f"\n   💡 建议:")
            print(f"      1. 检查残基编号是否正确")
            print(f"      2. 使用以下命令查看文件中的实际残基编号:")
            print(f"         grep '^ATOM' {pdbqt_file} | awk '{{print $5}}' | sort -u")
            return False, msg
        
        print(f"   ✅ 所有残基都存在")
        return True, "OK"
        
    except Exception as e:
        return False, f"检查失败: {str(e)}"


def make_flexible_verbose(base_pdbqt: str, filename: str, 
                         output_dir: str, flexible_residues: str,
                         verbose: bool = True) -> Tuple[List[str], Optional[str]]:
    """
    创建柔性受体 - 带详细调试信息
    
    Args:
        base_pdbqt: 输入的完整受体 PDBQT
        filename: 文件基本名
        output_dir: 输出目录
        flexible_residues: 柔性残基 (如 "235:102:157")
        verbose: 详细输出
    
    Returns:
        ([成功的文件列表], 错误消息)
    """
    
    print(f"\n{'='*60}")
    print(f"🔧 开始柔性处理")
    print(f"{'='*60}")
    print(f"输入文件: {base_pdbqt}")
    print(f"柔性残基: {flexible_residues}")
    
    # 1. 检查输入文件
    if not os.path.exists(base_pdbqt):
        return [], f"输入文件不存在: {base_pdbqt}"
    
    file_size = os.path.getsize(base_pdbqt)
    print(f"✅ 输入文件存在 ({file_size} bytes)")
    
    # 2. 检查残基是否存在
    residues_ok, residues_msg = check_residues_in_pdbqt(base_pdbqt, flexible_residues)
    if not residues_ok:
        return [], f"残基检查失败: {residues_msg}"
    
    # 3. 准备输出文件路径
    rigid_pdbqt = os.path.join(output_dir, f"{filename}_rigid.pdbqt")
    flex_pdbqt = os.path.join(output_dir, f"{filename}_flex.pdbqt")
    
    # 重命名输入文件为 rigid
    print(f"\n📝 重命名输入文件...")
    print(f"   {base_pdbqt}")
    print(f"   → {rigid_pdbqt}")
    os.rename(base_pdbqt, rigid_pdbqt)
    
    # 4. 构建命令
    cmd = [
        Config.MGLTOOLS_PATH,
        Config.PREPARE_FLEXRECEPTOR,
        "-r", rigid_pdbqt,
        "-s", flexible_residues,
        "-o", rigid_pdbqt,  # 更新的刚性部分（覆盖）
        "-x", flex_pdbqt    # 柔性部分
    ]
    
    print(f"\n🚀 执行 prepare_flexreceptor4.py...")
    print(f"命令: {' '.join(cmd)}")
    
    # 5. 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = Config.PYTHONPATH
    
    # 6. 执行命令
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=Config.TIMEOUT_SECONDS
        )
        
        print(f"\n📋 执行结果:")
        print(f"   返回码: {result.returncode}")
        
        # 显示详细输出
        if result.stdout:
            print(f"\n📤 标准输出:")
            for line in result.stdout.strip().split('\n'):
                print(f"   {line}")
        
        if result.stderr:
            print(f"\n⚠️  标准错误:")
            for line in result.stderr.strip().split('\n'):
                print(f"   {line}")
        
        # 7. 检查输出文件
        print(f"\n📁 检查输出文件:")
        
        rigid_exists = os.path.exists(rigid_pdbqt)
        flex_exists = os.path.exists(flex_pdbqt)
        
        if rigid_exists:
            rigid_size = os.path.getsize(rigid_pdbqt)
            print(f"   ✅ Rigid: {rigid_pdbqt} ({rigid_size} bytes)")
            
            # 统计原子数
            with open(rigid_pdbqt, 'r') as f:
                rigid_lines = f.readlines()
            rigid_atoms = sum(1 for line in rigid_lines if line.startswith(('ATOM', 'HETATM')))
            print(f"      原子数: {rigid_atoms}")
        else:
            print(f"   ❌ Rigid 文件未生成")
        
        if flex_exists:
            flex_size = os.path.getsize(flex_pdbqt)
            print(f"   ✅ Flex: {flex_pdbqt} ({flex_size} bytes)")
            
            # 统计柔性信息
            with open(flex_pdbqt, 'r') as f:
                flex_lines = f.readlines()
            flex_atoms = sum(1 for line in flex_lines if line.startswith(('ATOM', 'HETATM')))
            flex_branches = sum(1 for line in flex_lines if line.startswith('BRANCH'))
            print(f"      原子数: {flex_atoms}")
            print(f"      可旋转键数: {flex_branches}")
            
            # 显示部分内容
            if verbose:
                print(f"\n   📄 Flex 文件前15行:")
                for i, line in enumerate(flex_lines[:15], 1):
                    print(f"      {i:2d}: {line.rstrip()}")
        else:
            print(f"   ❌ Flex 文件未生成")
        
        # 8. 判断成功与否
        if result.returncode == 0 and flex_exists:
            print(f"\n{'='*60}")
            print(f"✅ 柔性处理成功!")
            print(f"{'='*60}")
            return [rigid_pdbqt, flex_pdbqt], None
        
        else:
            # 失败情况
            error_msg = "Flex 文件未生成"
            if result.stderr:
                error_msg = result.stderr.strip()[:300]
            
            print(f"\n{'='*60}")
            print(f"❌ 柔性处理失败")
            print(f"{'='*60}")
            print(f"错误: {error_msg}")
            
            # 提供故障排除建议
            print(f"\n💡 故障排除建议:")
            print(f"1. 检查残基编号格式")
            print(f"   当前: {flexible_residues}")
            print(f"   格式应该是: 235:102:157 (用冒号分隔)")
            print(f"")
            print(f"2. 如果有多条链，尝试指定链ID")
            print(f"   格式: A:235:A:102:A:157")
            print(f"")
            print(f"3. 检查残基是否真实存在")
            print(f"   运行: grep '^ATOM' {rigid_pdbqt} | head -20")
            print(f"")
            print(f"4. 尝试单个残基测试")
            print(f"   先用一个残基测试: --flex-res 235")
            
            # 保留 rigid 文件
            if rigid_exists:
                print(f"\n✅ 保留刚性受体文件: {rigid_pdbqt}")
                return [rigid_pdbqt], error_msg
            else:
                return [], error_msg
    
    except subprocess.TimeoutExpired:
        error = f"超时 (>{Config.TIMEOUT_SECONDS}秒)"
        print(f"❌ {error}")
        return [rigid_pdbqt] if os.path.exists(rigid_pdbqt) else [], error
    
    except Exception as e:
        error = f"异常: {str(e)}"
        print(f"❌ {error}")
        import traceback
        traceback.print_exc()
        return [rigid_pdbqt] if os.path.exists(rigid_pdbqt) else [], error


def test_flexible_processing(pdb_file: str, output_dir: str, 
                            flexible_residues: str):
    """
    测试柔性处理的完整流程
    """
    from pathlib import Path
    
    print(f"\n{'='*60}")
    print(f"测试柔性对接预处理")
    print(f"{'='*60}")
    print(f"PDB 文件: {pdb_file}")
    print(f"输出目录: {output_dir}")
    print(f"柔性残基: {flexible_residues}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = Path(pdb_file).stem
    
    # Step 1: 生成基础 PDBQT (使用 prepare_receptor4.py)
    print(f"\n{'='*60}")
    print(f"步骤 1: 生成基础 PDBQT")
    print(f"{'='*60}")
    
    base_pdbqt = os.path.join(output_dir, f"{filename}.pdbqt")
    
    cmd = [
        Config.MGLTOOLS_PATH,
        "/usr/local/autodocktools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
        "-r", pdb_file,
        "-o", base_pdbqt,
        "-A", "hydrogens",
        "-U", "nphs_lps_waters"
    ]
    
    print(f"命令: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = Config.PYTHONPATH
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    
    if result.returncode != 0 or not os.path.exists(base_pdbqt):
        print(f"❌ 基础 PDBQT 生成失败")
        if result.stderr:
            print(f"错误: {result.stderr}")
        return False
    
    print(f"✅ 基础 PDBQT 生成成功: {base_pdbqt}")
    
    # Step 2: 创建柔性受体
    output_files, error = make_flexible_verbose(
        base_pdbqt,
        filename,
        output_dir,
        flexible_residues,
        verbose=True
    )
    
    if len(output_files) == 2:
        print(f"\n{'='*60}")
        print(f"✅✅✅ 测试成功! ✅✅✅")
        print(f"{'='*60}")
        print(f"生成的文件:")
        for f in output_files:
            print(f"   - {f}")
        return True
    else:
        print(f"\n{'='*60}")
        print(f"❌ 测试失败")
        print(f"{'='*60}")
        if error:
            print(f"错误: {error}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试和调试柔性对接处理')
    parser.add_argument('--pdb', required=True, help='输入 PDB 文件')
    parser.add_argument('--flex-res', required=True, help='柔性残基 (如: 235:102:157)')
    parser.add_argument('--output', default='test_output', help='输出目录')
    
    args = parser.parse_args()
    
    success = test_flexible_processing(args.pdb, args.output, args.flex_res)
    
    import sys
    sys.exit(0 if success else 1)
