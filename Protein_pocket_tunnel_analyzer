#!/usr/bin/env python3
"""
Protein Structure Analyzer - Batch Processing of protein pockets and tunnels

"""

import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import shutil
import re


class ProteinAnalyzer:
    """蛋白质结构分析 - 整合P2Rank和MOLE2"""
    
    def __init__(self, work_dir: str = "/content/protein_analysis"):
        """
        初始化分析器
        
        Args:
            work_dir: 工作目录路径
        """
        self.work_dir = Path(work_dir)
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        self.p2rank_output = self.output_dir / "p2rank"
        self.mole2_output = self.output_dir / "mole2"
        
        # 工具路径
        self.p2rank_path = self.work_dir / "p2rank_2.4.2"
        self.mole2_path = self.work_dir / "mole2"
        
        # 创建目录
        self._create_directories()
        
    def _create_directories(self):
        """创建必要的工作目录"""
        for directory in [self.input_dir, self.output_dir, 
                         self.p2rank_output, self.mole2_output]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def setup_environment(self):
        """安装和配置P2Rank和MOLE2环境"""
        print("=" * 60)
        print("🔧 Setting up environment...")
        print("=" * 60)
        
        # 1. 安装Java
        self._install_java()
        
        # 2. 下载P2Rank
        self._download_p2rank()
        
        # 3. 下载MOLE2
        self._download_mole2()
        
        print("\n✅ Environment setup completed!")
        print("=" * 60)
    
    def _install_java(self):
        """安装Java运行环境"""
        print("\n☕ Installing Java...")
        try:
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                stderr=subprocess.STDOUT
            )
            if result.returncode == 0:
                print("✅ Java already installed")
                return
        except FileNotFoundError:
            pass
        
        # 安装OpenJDK
        subprocess.run(
            ["apt-get", "update", "-qq"],
            check=True,
            stdout=subprocess.DEVNULL
        )
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "openjdk-11-jdk"],
            check=True,
            stdout=subprocess.DEVNULL
        )
        print("✅ Java installed successfully")
    
    def _download_p2rank(self):
        """下载P2Rank"""
        print("\n📦 Downloading P2Rank...")
        
        if self.p2rank_path.exists():
            print("✅ P2Rank already exists")
            return
        
        # 下载最新版本
        p2rank_url = "https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz"
        
        subprocess.run(
            ["wget", "-q", "-O", f"{self.work_dir}/p2rank.tar.gz", p2rank_url],
            check=True
        )
        
        # 解压
        subprocess.run(
            ["tar", "-xzf", f"{self.work_dir}/p2rank.tar.gz", "-C", str(self.work_dir)],
            check=True
        )
        
        # 删除压缩包
        os.remove(f"{self.work_dir}/p2rank.tar.gz")
        
        print("✅ P2Rank downloaded and extracted")
    
    def _download_mole2(self):
        """下载MOLE2命令行版本"""
        print("\n📦 Downloading MOLE2...")
        
        mole2_jar = self.mole2_path / "Mole2.jar"
        
        if mole2_jar.exists():
            print("✅ MOLE2 already exists")
            return
        
        self.mole2_path.mkdir(exist_ok=True)
        
        # MOLE2命令行版本下载链接
        mole2_url = "https://webchem.ncbr.muni.cz/Platform/AppsBin/Mole/2.5.24.6.8/Mole2_cmd.zip"
        mole2_zip = self.mole2_path / "mole2_cmd.zip"
        
        # 下载ZIP文件
        subprocess.run(
            ["wget", "-q", "-O", str(mole2_zip), mole2_url],
            check=True
        )
        
        # 解压
        subprocess.run(
            ["unzip", "-q", str(mole2_zip), "-d", str(self.mole2_path)],
            check=True
        )
        
        # 删除ZIP文件
        mole2_zip.unlink()
        
        # 验证JAR文件是否存在
        if not mole2_jar.exists():
            # 尝试查找解压后的JAR文件
            jar_files = list(self.mole2_path.rglob("*.jar"))
            if jar_files:
                # 将找到的JAR文件移动到标准位置
                shutil.move(str(jar_files[0]), str(mole2_jar))
        
        print("✅ MOLE2 downloaded and extracted")
    
    def run_p2rank(self, pdb_files: List[str] = None) -> Dict[str, Dict]:
        """
        批量运行P2Rank分析
        
        Args:
            pdb_files: PDB文件列表，None则处理input目录所有PDB
            
        Returns:
            Dict[pdb_name, pocket_info]: 每个蛋白的口袋信息
        """
        print("\n" + "=" * 60)
        print("🔍 Running P2Rank Analysis...")
        print("=" * 60)
        
        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]
        
        if not pdb_files:
            print("⚠️  No PDB files found!")
            return {}
        
        results = {}
        
        for pdb_file in pdb_files:
            print(f"\n📊 Analyzing {pdb_file.name}...")
            
            # 运行P2Rank
            prank_script = self.p2rank_path / "prank"
            output_subdir = self.p2rank_output / pdb_file.stem
            
            cmd = [
                str(prank_script),
                "predict",
                str(pdb_file),
                "-o", str(output_subdir),
                "-threads", "2"
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"✅ P2Rank completed for {pdb_file.name}")
                    
                    # 解析结果
                    pocket_info = self._parse_p2rank_results(output_subdir, pdb_file.stem)
                    results[pdb_file.stem] = pocket_info
                    
                else:
                    print(f"❌ P2Rank failed for {pdb_file.name}")
                    print(f"Error: {result.stderr}")
                    results[pdb_file.stem] = {"error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                print(f"⏱️  Timeout for {pdb_file.name}")
                results[pdb_file.stem] = {"error": "Timeout"}
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results[pdb_file.stem] = {"error": str(e)}
        
        print("\n" + "=" * 60)
        print("✅ P2Rank analysis completed!")
        print("=" * 60)
        
        return results
    
    def _parse_p2rank_results(self, output_dir: Path, pdb_name: str) -> Dict:
        """解析P2Rank输出结果"""
        pocket_info = {
            "pdb_name": pdb_name,
            "pockets": [],
            "summary": {}
        }
        
        # 查找CSV结果文件
        csv_pattern = output_dir / "*.pdb_predictions.csv"
        csv_files = list(output_dir.glob(f"{pdb_name}.pdb_predictions.csv"))
        
        if not csv_files:
            # 尝试其他可能的文件名模式
            csv_files = list(output_dir.glob("*.csv"))
        
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                
                pocket_info["summary"] = {
                    "total_pockets": len(df),
                    "output_file": str(csv_files[0])
                }
                
                # 提取口袋信息
                for idx, row in df.iterrows():
                    pocket = {
                        "rank": int(row.get("rank", idx + 1)),
                        "score": float(row.get("score", 0)),
                        "probability": float(row.get("probability", 0)),
                        "sas_points": int(row.get("sas_points", 0)),
                        "center_x": float(row.get("center_x", 0)),
                        "center_y": float(row.get("center_y", 0)),
                        "center_z": float(row.get("center_z", 0)),
                    }
                    
                    # 添加残基信息（如果有）
                    if "residue_ids" in row:
                        pocket["residues"] = str(row["residue_ids"])
                    
                    pocket_info["pockets"].append(pocket)
                
            except Exception as e:
                pocket_info["parse_error"] = str(e)
        
        return pocket_info
    
    def run_mole2(self, pdb_files: List[str] = None, 
                  use_p2rank_pockets: bool = True) -> Dict[str, Dict]:
        """
        批量运行MOLE2分析
        
        Args:
            pdb_files: PDB文件列表
            use_p2rank_pockets: 是否使用P2Rank检测的口袋作为起点
            
        Returns:
            Dict[pdb_name, tunnel_info]: 每个蛋白的通道信息
        """
        print("\n" + "=" * 60)
        print("🌀 Running MOLE2 Analysis...")
        print("=" * 60)
        
        if pdb_files is None:
            pdb_files = list(self.input_dir.glob("*.pdb"))
        else:
            pdb_files = [Path(f) for f in pdb_files]
        
        if not pdb_files:
            print("⚠️  No PDB files found!")
            return {}
        
        results = {}
        
        # 查找MOLE2 JAR文件
        mole2_jar = self.mole2_path / "Mole2.jar"
        if not mole2_jar.exists():
            # 尝试其他可能的文件名
            jar_files = list(self.mole2_path.glob("*.jar"))
            if jar_files:
                mole2_jar = jar_files[0]
            else:
                print("❌ MOLE2 JAR file not found!")
                return {}
        
        for pdb_file in pdb_files:
            print(f"\n🌀 Analyzing tunnels in {pdb_file.name}...")
            
            output_subdir = self.mole2_output / pdb_file.stem
            output_subdir.mkdir(exist_ok=True)
            
            # 构建MOLE2命令
            cmd = [
                "java", "-jar", str(mole2_jar),
                "-p", str(pdb_file),
                "-o", str(output_subdir)
            ]
            
            # 如果使用P2Rank口袋信息
            if use_p2rank_pockets:
                p2rank_result_dir = self.p2rank_output / pdb_file.stem
                pocket_pdb = list(p2rank_result_dir.glob("*_points.pdb"))
                
                if pocket_pdb:
                    cmd.extend(["-s", str(pocket_pdb[0])])
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"✅ MOLE2 completed for {pdb_file.name}")
                    
                    # 解析结果
                    tunnel_info = self._parse_mole2_results(output_subdir, pdb_file.stem)
                    results[pdb_file.stem] = tunnel_info
                    
                else:
                    print(f"❌ MOLE2 failed for {pdb_file.name}")
                    print(f"Error: {result.stderr}")
                    results[pdb_file.stem] = {"error": result.stderr}
                    
            except subprocess.TimeoutExpired:
                print(f"⏱️  Timeout for {pdb_file.name}")
                results[pdb_file.stem] = {"error": "Timeout"}
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                results[pdb_file.stem] = {"error": str(e)}
        
        print("\n" + "=" * 60)
        print("✅ MOLE2 analysis completed!")
        print("=" * 60)
        
        return results
    
    def _parse_mole2_results(self, output_dir: Path, pdb_name: str) -> Dict:
        """解析MOLE2输出结果"""
        tunnel_info = {
            "pdb_name": pdb_name,
            "tunnels": [],
            "summary": {}
        }
        
        # 查找输出文件
        result_files = list(output_dir.glob("*.xml")) + list(output_dir.glob("*.json"))
        
        if result_files:
            tunnel_info["summary"] = {
                "output_files": [str(f) for f in result_files],
                "tunnel_count": len(list(output_dir.glob("Tunnel_*")))
            }
        
        return tunnel_info
    
    def generate_summary_report(self, p2rank_results: Dict, 
                               mole2_results: Dict) -> pd.DataFrame:
        """
        生成综合分析报告
        
        Args:
            p2rank_results: P2Rank分析结果
            mole2_results: MOLE2分析结果
            
        Returns:
            DataFrame: 汇总报告
        """
        print("\n" + "=" * 60)
        print("📋 Generating Summary Report...")
        print("=" * 60)
        
        report_data = []
        
        all_proteins = set(p2rank_results.keys()) | set(mole2_results.keys())
        
        for protein in all_proteins:
            p2rank_data = p2rank_results.get(protein, {})
            mole2_data = mole2_results.get(protein, {})
            
            row = {
                "Protein": protein,
                "Total_Pockets": p2rank_data.get("summary", {}).get("total_pockets", 0),
                "Top_Pocket_Score": 0,
                "Top_Pocket_Probability": 0,
                "Tunnel_Count": mole2_data.get("summary", {}).get("tunnel_count", 0),
                "P2Rank_Status": "Success" if "error" not in p2rank_data else "Failed",
                "MOLE2_Status": "Success" if "error" not in mole2_data else "Failed"
            }
            
            # 获取最高评分口袋信息
            if p2rank_data.get("pockets"):
                top_pocket = p2rank_data["pockets"][0]
                row["Top_Pocket_Score"] = top_pocket.get("score", 0)
                row["Top_Pocket_Probability"] = top_pocket.get("probability", 0)
            
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # 保存报告
        report_path = self.output_dir / "summary_report.csv"
        df.to_csv(report_path, index=False)
        
        print(f"\n✅ Summary report saved to: {report_path}")
        print("\n" + "=" * 60)
        
        return df
    
    def save_detailed_results(self, p2rank_results: Dict, mole2_results: Dict):
        """保存详细的JSON结果"""
        results = {
            "p2rank": p2rank_results,
            "mole2": mole2_results
        }
        
        json_path = self.output_dir / "detailed_results.json"
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Detailed results saved to: {json_path}")


def main():
    """主函数 - 用于测试"""
    analyzer = ProteinAnalyzer()
    
    # 设置环境
    analyzer.setup_environment()
    
    # 运行分析
    p2rank_results = analyzer.run_p2rank()
    mole2_results = analyzer.run_mole2(use_p2rank_pockets=True)
    
    # 生成报告
    summary_df = analyzer.generate_summary_report(p2rank_results, mole2_results)
    analyzer.save_detailed_results(p2rank_results, mole2_results)
    
    return analyzer, summary_df


if __name__ == "__main__":
    analyzer, summary = main()
