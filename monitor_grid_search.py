#!/usr/bin/env python3
"""
网格搜索进度监控脚本
实时监控模型训练进度，包括网格搜索状态
"""

import psutil
import time
import datetime
import os
import re
import json
from pathlib import Path

class GridSearchMonitor:
    def __init__(self, pid=None, log_file="retrain_final.log"):
        self.pid = pid
        self.log_file = log_file
        self.start_time = time.time()
        self.last_log_size = 0
        self.grid_search_start = None
        self.cv_results = []
        
    def check_process_status(self):
        """检查进程状态"""
        if not self.pid or not psutil.pid_exists(self.pid):
            return {"status": "not_found", "message": "进程不存在"}
            
        try:
            p = psutil.Process(self.pid)
            return {
                "status": p.status(),
                "cpu_percent": p.cpu_percent(),
                "memory_mb": p.memory_info().rss / 1024 / 1024,
                "runtime": str(datetime.datetime.now() - datetime.datetime.fromtimestamp(p.create_time()))
            }
        except psutil.NoSuchProcess:
            return {"status": "terminated", "message": "进程已终止"}
    
    def parse_log_progress(self):
        """解析日志文件获取进度信息"""
        if not os.path.exists(self.log_file):
            return {"error": "日志文件不存在"}
            
        try:
            current_size = os.path.getsize(self.log_file)
            if current_size == self.last_log_size:
                return {"status": "no_update"}
            
            self.last_log_size = current_size
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            progress = {
                "log_size": current_size,
                "last_update": datetime.datetime.fromtimestamp(os.path.getmtime(self.log_file)).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 解析批次处理进度
            batch_matches = re.findall(r'处理批次 (\d+)/(\d+)', content)
            if batch_matches:
                current_batch = max(int(match[0]) for match in batch_matches)
                total_batches = int(batch_matches[0][1])
                progress["batch_progress"] = f"{current_batch}/{total_batches} ({current_batch/total_batches*100:.1f}%)"
            
            # 解析数据生成进度
            feature_matches = re.findall(r'生成 (\d+) 条特征样本', content)
            if feature_matches:
                total_features = sum(int(match) for match in feature_matches)
                progress["total_features"] = f"{total_features:,}"
            
            # 解析合并后数据
            merged_matches = re.findall(r'合并后样本数量[:：]\s*(\d+)', content)
            if merged_matches:
                progress["merged_samples"] = f"{int(merged_matches[-1]):,}"
            
            # 解析网格搜索开始
            if "开始网格搜索" in content and not self.grid_search_start:
                self.grid_search_start = time.time()
                progress["grid_search_started"] = True
            
            # 解析交叉验证进度
            cv_matches = re.findall(r'CV得分[:：]\s*([\d.]+)', content)
            if cv_matches:
                progress["cv_scores"] = [float(score) for score in cv_matches[-5:]]  # 最近5个得分
            
            # 解析最优参数
            param_matches = re.findall(r'最优参数[:：]\s*({[^}]+})', content)
            if param_matches:
                try:
                    params = eval(param_matches[-1])
                    progress["best_params"] = params
                except:
                    progress["best_params"] = param_matches[-1]
            
            # 解析模型训练完成
            if "训练集AUC" in content and "测试集AUC" in content:
                progress["model_training_complete"] = True
                
                # 提取AUC分数
                auc_matches = re.findall(r'(训练集|测试集)AUC[:：]\s*([\d.]+)', content)
                if auc_matches:
                    progress["auc_scores"] = {match[0]: float(match[1]) for match in auc_matches}
            
            return progress
            
        except Exception as e:
            return {"error": f"解析日志失败: {e}"}
    
    def estimate_grid_search_progress(self):
        """估算网格搜索进度"""
        if not self.grid_search_start:
            return {"status": "not_started"}
        
        elapsed_time = time.time() - self.grid_search_start
        
        # 基于经验估算：8个参数组合 × 5折交叉验证 = 40个拟合
        total_fits = 40
        estimated_time_per_fit = 30  # 每个拟合约30秒
        total_estimated_time = total_fits * estimated_time_per_fit
        
        progress_ratio = min(elapsed_time / total_estimated_time, 0.95)  # 最多显示95%
        remaining_time = max(total_estimated_time - elapsed_time, 0)
        
        return {
            "elapsed_time": str(datetime.timedelta(seconds=int(elapsed_time))),
            "estimated_progress": f"{progress_ratio*100:.1f}%",
            "estimated_remaining": str(datetime.timedelta(seconds=int(remaining_time))),
            "total_fits": total_fits,
            "completed_fits": int(total_fits * progress_ratio)
        }
    
    def get_model_files_status(self):
        """检查模型文件状态"""
        models_dir = Path("models/enhanced")
        if not models_dir.exists():
            return {"status": "no_models_dir"}
        
        model_files = list(models_dir.glob("*.pkl"))
        json_files = list(models_dir.glob("*.json"))
        
        return {
            "model_files": len(model_files),
            "json_files": len(json_files),
            "latest_model": max(model_files, key=lambda x: x.stat().st_mtime).name if model_files else None,
            "latest_json": max(json_files, key=lambda x: x.stat().st_mtime).name if json_files else None
        }
    
    def print_status(self):
        """打印当前状态"""
        print("\n" + "="*60)
        print(f"网格搜索进度监控 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # 进程状态
        process_status = self.check_process_status()
        print(f"进程状态: {process_status}")
        
        # 日志进度
        log_progress = self.parse_log_progress()
        print(f"\n日志进度: {log_progress}")
        
        # 网格搜索估算
        if self.grid_search_start or "开始网格搜索" in str(log_progress):
            grid_progress = self.estimate_grid_search_progress()
            print(f"\n网格搜索进度: {grid_progress}")
        
        # 模型文件状态
        model_status = self.get_model_files_status()
        print(f"\n模型文件: {model_status}")
        
        print("\n" + "="*60)

def main():
    """主函数：持续监控网格搜索进度"""
    import argparse
    
    parser = argparse.ArgumentParser(description="网格搜索进度监控工具")
    parser.add_argument("--pid", type=int, help="进程ID")
    parser.add_argument("--log", default="retrain_final.log", help="日志文件路径")
    parser.add_argument("--interval", type=int, default=30, help="监控间隔(秒)")
    parser.add_argument("--once", action="store_true", help="只监控一次")
    
    args = parser.parse_args()
    
    # 如果没有指定PID，尝试自动查找
    if not args.pid:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'retrain_optimized_models.py' in cmdline:
                    args.pid = proc.info['pid']
                    print(f"自动找到进程: {args.pid}")
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    if not args.pid:
        print("未找到训练进程，请手动指定 --pid 参数")
        return
    
    monitor = GridSearchMonitor(pid=args.pid, log_file=args.log)
    
    try:
        if args.once:
            monitor.print_status()
        else:
            print(f"开始监控进程 {args.pid}，按 Ctrl+C 停止...")
            while True:
                monitor.print_status()
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n监控停止")

if __name__ == "__main__":
    main()