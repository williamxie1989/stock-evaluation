#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastAPI 远程模型训练服务
端点：
POST /train               创建训练任务，返回 job_id
GET  /train/status/{job_id} 查询任务状态
GET  /download/{job_id}   下载训练完成后的模型压缩包

该服务直接使用项目现有的 UnifiedModelTrainer。
任务在后台线程中执行，状态缓存于内存字典（生产环境可替换为 Redis）。
"""
from __future__ import annotations

import os
import uuid
import shutil
import zipfile
import logging
from threading import Thread, Lock
from typing import Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 动态导入项目根目录
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.ml.training.v1.train_unified_models import UnifiedModelTrainer  # noqa: E402

app = FastAPI(title="Remote Training Service", version="0.1.0")

# ---------------------------- 数据结构 ---------------------------- #
class TrainRequest(BaseModel):
    mode: str = "both"              # classification / regression / both
    lookback_days: int = 365
    n_stocks: int = 1000
    prediction_period: int = 30
    enable_feature_selection: bool = True

class JobState(BaseModel):
    state: str  # pending / running / finished / failed
    progress: float = 0.0
    detail: str | None = None
    model_path: str | None = None  # zip 文件完整路径

# ---------------------------- 全局任务字典 ---------------------------- #
_JOBS: Dict[str, JobState] = {}
_LOCK = Lock()

# 模型输出根目录
OUTPUT_ROOT = ROOT_DIR / "models" / "remote"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------- 后台训练逻辑 ---------------------------- #

def _run_training(job_id: str, req: TrainRequest):
    trainer = UnifiedModelTrainer(enable_feature_selection=req.enable_feature_selection)
    with _LOCK:
        _JOBS[job_id].state = "running"
    try:
        X, y = trainer.prepare_training_data(
            mode=req.mode,
            lookback_days=req.lookback_days,
            n_stocks=req.n_stocks,
            prediction_period=req.prediction_period,
        )
        results = trainer.train_models(X, y, mode=req.mode)

        # 将结果模型目录压缩
        job_dir = OUTPUT_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # 假设 trainer 内部已有保存模型到 models 目录逻辑；此处仅演示写一份 results.pkl
        import pickle
        with open(job_dir / "train_results.pkl", "wb") as f:
            pickle.dump(results, f)

        zip_path = OUTPUT_ROOT / f"{job_id}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in job_dir.rglob('*'):
                zf.write(file_path, arcname=file_path.relative_to(job_dir))

        with _LOCK:
            _JOBS[job_id].state = "finished"
            _JOBS[job_id].model_path = str(zip_path)
            _JOBS[job_id].progress = 1.0
    except Exception as e:
        logger.exception("训练任务失败")
        with _LOCK:
            _JOBS[job_id].state = "failed"
            _JOBS[job_id].detail = str(e)

# ---------------------------- API 端点 ---------------------------- #

@app.post("/train")
async def create_train_job(req: TrainRequest):
    job_id = uuid.uuid4().hex
    # 初始化状态
    with _LOCK:
        _JOBS[job_id] = JobState(state="pending", progress=0.0)

    # 启动线程执行
    thread = Thread(target=_run_training, args=(job_id, req), daemon=True)
    thread.start()
    logger.info("启动训练任务 %s", job_id)
    return {"job_id": job_id}

@app.get("/train/status/{job_id}")
async def get_job_status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job.dict()

@app.get("/download/{job_id}")
async def download_model(job_id: str):
    job = _JOBS.get(job_id)
    if job is None or job.state != "finished" or not job.model_path:
        raise HTTPException(status_code=400, detail="模型尚未准备好")
    if not os.path.exists(job.model_path):
        raise HTTPException(status_code=500, detail="模型文件丢失")
    return FileResponse(job.model_path, filename=os.path.basename(job.model_path))

# 健康检查
@app.get("/ping")
async def ping():
    return JSONResponse({"msg": "pong"})