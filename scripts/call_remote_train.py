#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地脚本：触发远端模型训练并在完成后下载模型压缩包"""
import os
import time
import argparse
import requests
from urllib.parse import urljoin
from pathlib import Path
from dotenv import load_dotenv

# 读取 .env
load_dotenv()
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "http://127.0.0.1:8000/")
DOWNLOAD_DIR = Path(os.getenv("MODEL_DOWNLOAD_DIR", "models/remote_downloads"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Trigger remote training and download model")
    parser.add_argument("--mode", default="both", choices=["classification", "regression", "both"])
    parser.add_argument("--lookback_days", type=int, default=365)
    parser.add_argument("--n_stocks", type=int, default=1000)
    parser.add_argument("--prediction_period", type=int, default=30)
    parser.add_argument("--enable_feature_selection", action="store_true")
    parser.add_argument("--poll_interval", type=int, default=30, help="status 查询间隔秒数")
    return parser.parse_args()

def main():
    args = parse_args()
    payload = {
        "mode": args.mode,
        "lookback_days": args.lookback_days,
        "n_stocks": args.n_stocks,
        "prediction_period": args.prediction_period,
        "enable_feature_selection": args.enable_feature_selection,
    }

    # 1. 触发训练
    train_url = urljoin(REMOTE_API_URL, "train")
    resp = requests.post(train_url, json=payload, timeout=60)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"已创建训练任务 {job_id}")

    # 2. 轮询状态
    status_url = urljoin(REMOTE_API_URL, f"train/status/{job_id}")
    while True:
        time.sleep(args.poll_interval)
        status_resp = requests.get(status_url, timeout=30)
        if status_resp.status_code != 200:
            print("状态查询失败", status_resp.text)
            continue
        status = status_resp.json()
        print(f"进度: {status['state']} {status['progress']*100:.1f}%")
        if status["state"] == "finished":
            break
        if status["state"] == "failed":
            raise RuntimeError(f"远端训练失败: {status.get('detail')}")

    # 3. 下载模型
    download_url = urljoin(REMOTE_API_URL, f"download/{job_id}")
    dl_resp = requests.get(download_url, stream=True)
    dl_resp.raise_for_status()
    file_path = DOWNLOAD_DIR / f"{job_id}.zip"
    with open(file_path, "wb") as f:
        for chunk in dl_resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"模型已下载到 {file_path}")

if __name__ == "__main__":
    main()