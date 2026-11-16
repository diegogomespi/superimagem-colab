"""Backend CUDA do SuperImagem para Google Colab.

Mantém o contrato HTTP usado pela interface React. Os arquivos ficam somente
no diretório temporário da sessão do Colab.
"""
from __future__ import annotations

import asyncio
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# BasicSR antigo importa um módulo renomeado nas versões novas do torchvision.
from spandrel import ModelLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(tempfile.mkdtemp(prefix='superimagem-t4-'))
ARTIFACTS_DIR = DATA / 'artifacts'
MODELS_DIR = ROOT / 'colab' / 'models'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    'general': {
        'name': 'RealESRGAN_x4plus',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth',
        'file': 'RealESRGAN_x4plus.pth',
        'blocks': 23,
    },
    'illustration': {
        'name': 'RealESRGAN_x4plus_anime_6B',
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth',
        'file': 'RealESRGAN_x4plus_anime_6B.pth',
        'blocks': 6,
    },
}

app = FastAPI(title='SuperImagem GPU T4', version='1.0.0')
artifacts: dict[str, dict[str, Any]] = {}
jobs: dict[str, dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='superimagem-gpu')
active_lock = threading.Lock()
models: dict[str, Any] = {}


class JobOptions(BaseModel):
    scale: int = 4
    upscaleModel: str = 'general'
    detailMode: str = 'faithful'


class JobRequest(BaseModel):
    artifactId: str
    operation: str = 'upscale'
    options: JobOptions = JobOptions()


def new_id(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex}'


def artifact_public(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != 'path'}


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 1_000_000:
        return
    import urllib.request
    partial = target.with_suffix(target.suffix + '.part')
    urllib.request.urlretrieve(url, partial)
    partial.replace(target)


def get_model(kind: str, job: dict[str, Any]) -> Any:
    if kind not in MODEL_SPECS:
        raise ValueError('Tipo de imagem inválido.')
    spec = MODEL_SPECS[kind]
    model_path = MODELS_DIR / spec['file']
    job['phase'] = 'loading_model'
    download(spec['url'], model_path)
    if kind not in models:
        descriptor = ModelLoader().load_from_file(str(model_path)).eval().cuda().half()
        if int(descriptor.scale) != 4:
            raise RuntimeError(f'Escala inesperada do modelo: {descriptor.scale}.')
        models[kind] = descriptor
    return models[kind]


def tile_count(width: int, height: int) -> int:
    return math.ceil(width / 400) * math.ceil(height / 400)


def extend_transparent_colors(bgra: np.ndarray) -> np.ndarray:
    """Estende as cores da borda para o vazio e evita halo preto/branco."""
    bgr = bgra[:, :, :3].copy()
    known = (bgra[:, :, 3] > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(10):
        expanded = cv2.dilate(known, kernel, iterations=1)
        frontier = (expanded > 0) & (known == 0)
        if not frontier.any():
            break
        for channel in range(3):
            spread = cv2.dilate(bgr[:, :, channel], kernel, iterations=1)
            bgr[:, :, channel][frontier] = spread[frontier]
        known = expanded
    return bgr


def upscale_rgb_tiled(bgr: np.ndarray, model: Any, job: dict[str, Any], completed: list[int], total: int) -> np.ndarray:
    height, width = bgr.shape[:2]
    native = 4
    output = np.empty((height * native, width * native, 3), dtype=np.uint8)
    tile, pad = 400, 24
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if job.get('cancelRequested'):
                raise InterruptedError('Processamento cancelado.')
            core_w, core_h = min(tile, width - x), min(tile, height - y)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(width, x + core_w + pad), min(height, y + core_h + pad)
            patch = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).unsqueeze(0).cuda().half().div_(255.0)
            with torch.inference_mode():
                prediction = model(tensor).clamp_(0, 1)
            result = prediction[0].float().cpu().numpy().transpose(1, 2, 0)
            result = cv2.cvtColor(np.rint(result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            crop_x, crop_y = (x - x0) * native, (y - y0) * native
            output[y * native:(y + core_h) * native, x * native:(x + core_w) * native] = result[crop_y:crop_y + core_h * native, crop_x:crop_x + core_w * native]
            completed[0] += 1
            job['progress']['completedTiles'] = min(completed[0], total)
            del tensor, prediction
    return output


def process_job(job_id: str) -> None:
    job = jobs[job_id]
    started = time.perf_counter()
    peak = psutil.Process().memory_info().rss
    try:
        with active_lock:
            if job.get('cancelRequested'):
                raise InterruptedError('Processamento cancelado.')
            source = artifacts[job['artifactId']]
            options = job['options']
            scale = int(options['scale'])
            if scale not in (2, 4, 8):
                raise ValueError('Use ampliação 2×, 4× ou 8×.')
            job.update(status='preparing', phase='preparing')
            raw = np.fromfile(source['path'], dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError('Não foi possível abrir a imagem.')
            has_alpha = image.ndim == 3 and image.shape[2] == 4
            height, width = image.shape[:2]
            detailed = scale == 8 and options.get('detailMode') == 'detailed'
            total = tile_count(width, height)
            if detailed:
                total += tile_count(width * 4, height * 4)
            job['progress'] = {'completedTiles': 0, 'totalTiles': total}
            model = get_model(options.get('upscaleModel', 'general'), job)
            completed = [0]
            job.update(status='processing', phase='inference')
            source_bgr = extend_transparent_colors(image) if has_alpha else image[:, :, :3]
            first4 = upscale_rgb_tiled(source_bgr, model, job, completed, total)
            if detailed:
                job['phase'] = 'second_pass'
                second4 = upscale_rgb_tiled(first4, model, job, completed, total)
                target_size = (width * 8, height * 8)
                rgb = cv2.resize(second4, target_size, interpolation=cv2.INTER_LANCZOS4)
                del second4
            else:
                target_size = (width * scale, height * scale)
                rgb = first4 if scale == 4 else cv2.resize(first4, target_size, interpolation=cv2.INTER_LANCZOS4)
            del first4
            if has_alpha:
                job['phase'] = 'alpha'
                alpha = cv2.resize(image[:, :, 3], target_size, interpolation=cv2.INTER_LANCZOS4)
                output = np.dstack((rgb, alpha))
            else:
                output = rgb

            job['phase'] = 'encoding'
            result_id = new_id('img')
            output_path = ARTIFACTS_DIR / f'{result_id}.png'
            ok, encoded = cv2.imencode('.png', output, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                raise RuntimeError('Falha ao gerar PNG.')
            encoded.tofile(output_path)
            out_height, out_width = output.shape[:2]
            result = {'id': result_id, 'filename': f'SuperImagem-{scale}x.png', 'mime': 'image/png', 'width': out_width, 'height': out_height, 'hasAlpha': output.ndim == 3 and output.shape[2] == 4, 'path': str(output_path), 'parentId': source['id']}
            artifacts[result_id] = result
            job.update(status='done', phase='done', resultArtifactId=result_id, progress={'completedTiles': total, 'totalTiles': total}, metrics={'durationMs': round((time.perf_counter() - started) * 1000), 'peakRssBytes': peak, 'modelId': MODEL_SPECS[options.get('upscaleModel', 'general')]['name']})
    except InterruptedError as exc:
        job.update(status='cancelled', phase='cancelled', error={'code': 'CANCELLED', 'message': str(exc)})
    except Exception as exc:
        if job.get('cancelRequested'):
            job.update(status='cancelled', phase='cancelled', error={'code': 'CANCELLED', 'message': 'Processamento cancelado.'})
        else:
            job.update(status='error', phase='error', error={'code': 'PROCESSING_ERROR', 'message': str(exc)})


@app.get('/api/health')
def health() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {'name': props.name, 'memoryBytes': props.total_memory, 'cuda': torch.version.cuda}
    return {'name': 'SuperImagem', 'version': '1.0.0-colab', 'engineAvailable': torch.cuda.is_available(), 'engine': 'cuda' if torch.cuda.is_available() else 'cpu', 'gpu': gpu, 'workerBusy': any(j['status'] in ('queued', 'preparing', 'processing') for j in jobs.values())}


@app.post('/api/artifacts')
async def upload_artifact(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in ('image/png', 'image/jpeg', 'image/webp'):
        raise HTTPException(415, 'Use PNG, JPEG ou WebP.')
    data = await file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, 'O arquivo excede 100 MB.')
    item_id = new_id('img')
    suffix = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}[file.content_type]
    path = ARTIFACTS_DIR / f'{item_id}{suffix}'
    path.write_bytes(data)
    try:
        with Image.open(path) as image:
            width, height = image.size
            has_alpha = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, 'Imagem inválida.')
    item = {'id': item_id, 'filename': file.filename or f'imagem{suffix}', 'mime': file.content_type, 'width': width, 'height': height, 'hasAlpha': has_alpha, 'path': str(path)}
    artifacts[item_id] = item
    return artifact_public(item)


@app.post('/api/jobs', status_code=202)
async def create_job(request: JobRequest) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise HTTPException(503, 'GPU CUDA não encontrada. No Colab, selecione um ambiente T4.')
    if request.operation != 'upscale' or request.artifactId not in artifacts:
        raise HTTPException(400, 'Imagem ou operação inválida.')
    if any(j['status'] in ('queued', 'preparing', 'processing') for j in jobs.values()):
        raise HTTPException(409, 'Aguarde o trabalho atual terminar.')
    job_id = new_id('job')
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    job = {'id': job_id, 'artifactId': request.artifactId, 'operation': 'upscale', 'options': request.options.model_dump(), 'status': 'queued', 'phase': 'queued', 'progress': None, 'resultArtifactId': None, 'error': None, 'createdAt': now, 'updatedAt': now}
    jobs[job_id] = job
    asyncio.get_running_loop().run_in_executor(executor, process_job, job_id)
    return job


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(404, 'Trabalho não encontrado.')
    return jobs[job_id]


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str) -> dict[str, bool]:
    if job_id not in jobs:
        raise HTTPException(404, 'Trabalho não encontrado.')
    jobs[job_id]['cancelRequested'] = True
    jobs[job_id]['phase'] = 'cancelling'
    return {'ok': True}


@app.get('/api/artifacts/{artifact_id}/preview')
def preview(artifact_id: str) -> FileResponse:
    item = artifacts.get(artifact_id)
    if not item:
        raise HTTPException(404, 'Imagem não encontrada.')
    return FileResponse(item['path'], media_type=item['mime'])


@app.get('/api/artifacts/{artifact_id}/download')
def download_artifact(artifact_id: str, format: str = 'png', background: str = '#000000') -> FileResponse:
    item = artifacts.get(artifact_id)
    if not item:
        raise HTTPException(404, 'Imagem não encontrada.')
    if format == 'png':
        return FileResponse(item['path'], media_type='image/png', filename=f'SuperImagem-{artifact_id}.png')
    image = Image.open(item['path']).convert('RGBA')
    color = background.lstrip('#')
    rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4)) if len(color) == 6 else (0, 0, 0)
    flat = Image.new('RGB', image.size, rgb); flat.paste(image, mask=image.getchannel('A'))
    target = ARTIFACTS_DIR / f'{artifact_id}-black.jpg'; flat.save(target, quality=95, subsampling=0)
    return FileResponse(target, media_type='image/jpeg', filename=f'SuperImagem-{artifact_id}.jpg')


@app.on_event('shutdown')
def cleanup() -> None:
    shutil.rmtree(DATA, ignore_errors=True)


ui_root = ROOT / 'dist' / 'ui'
if ui_root.exists():
    app.mount('/', StaticFiles(directory=ui_root, html=True), name='ui')
