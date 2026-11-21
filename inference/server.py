#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 HTTP 生成服务（只用标准库 http.server，不用 FastAPI）。

接口：
  GET  /health   -> {"status": "ok"}
  POST /generate {"prompt": str, "max_new_tokens": int, "temperature": float,
                  "top_k": int, "top_p": float, "seed": int?}
       -> {"text": str, "gate": float | null}

gate 是提示的左支百分比（门控模型才有；非门控模型返回 null）。
服务是单模型常驻内存的 ThreadingHTTPServer；generate 调用串行化
（一把锁），因为模型本身不是线程安全的推理对象。

用法（模块）：
  from inference.server import create_app, run_server
  srv = create_app(model, tokenizer)
  srv.serve_forever()

用法（命令行，需要一个已转换的 .pt 检查点）：
  python -m inference.server --checkpoint checkpoints/generalist-v5_s600.pt \
      --host 127.0.0.1 --port 8000 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["create_app", "run_server"]


def _load_model_and_tokenizer(checkpoint: str, device: str):
    """从 .pt 检查点加载模型 + 词表（model/ 由 Agent A 提供）。"""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from evaluation._common import load_model_and_tokenizer  # noqa: E402
    return load_model_and_tokenizer(checkpoint, device)


def create_app(model, tokenizer, device=None):
    """返回一个已绑定模型与分词器的 ThreadingHTTPServer 子类实例的工厂。

    实际上返回 handler 类；run_server 用它建 server。直接用：
        server = ThreadingHTTPServer((host, port), create_app(model, tokenizer))
    """
    dev = device or _model_device(model)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 安静：不刷访问日志
            pass

        def _send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/health/"):
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/generate", "/generate/"):
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send_json(400, {"error": "bad Content-Length"})
                return
            try:
                req = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"bad json: {e}"})
                return
            prompt = req.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                self._send_json(400, {"error": "prompt must be a non-empty string"})
                return
            try:
                max_new = int(req.get("max_new_tokens", 100))
                temperature = float(req.get("temperature", 0.0))
                top_k = int(req.get("top_k", 0))
                top_p = float(req.get("top_p", 1.0))
                seed = req.get("seed")
                seed = int(seed) if seed is not None else None
            except (TypeError, ValueError) as e:
                self._send_json(400, {"error": f"bad parameters: {e}"})
                return
            if max_new < 0 or temperature < 0 or not (0.0 < top_p <= 1.0) or top_k < 0:
                self._send_json(400, {"error": "parameters out of range"})
                return
            try:
                from inference.generate import generate_with_gate
                with lock:
                    try:
                        text, gate = generate_with_gate(
                            model, tokenizer, prompt,
                            max_new_tokens=max_new, temperature=temperature,
                            top_k=top_k, top_p=top_p, device=dev, seed=seed)
                    except ValueError:
                        # 非门控模型：gate_values 抛错，只返回文本。
                        from inference.generate import generate
                        text = generate(
                            model, tokenizer, prompt,
                            max_new_tokens=max_new, temperature=temperature,
                            top_k=top_k, top_p=top_p, device=dev, seed=seed)
                        gate = None
            except Exception as e:
                self._send_json(500, {"error": f"generation failed: {e}"})
                return
            self._send_json(200, {"text": text, "gate": gate})

    return Handler


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cpu")


def run_server(host="127.0.0.1", port=8000, model=None, tokenizer=None,
               checkpoint=None, device="cpu"):
    """启动生成服务并一直跑下去（serve_forever）。

    model/tokenizer 直接给，或给 checkpoint 路径（此时现加载）。
    """
    if model is None or tokenizer is None:
        if not checkpoint:
            raise ValueError("run_server needs model+tokenizer or a checkpoint path")
        model, tokenizer = _load_model_and_tokenizer(checkpoint, device)
    handler = create_app(model, tokenizer, device=device)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"serving on http://{host}:{port}  (GET /health, POST /generate)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server


def main():
    ap = argparse.ArgumentParser(description="最小生成服务（stdlib http.server）")
    ap.add_argument("--checkpoint", required=True,
                    help="已转换的 .pt 检查点路径")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    try:
        run_server(args.host, args.port, checkpoint=args.checkpoint, device=args.device)
    except Exception as e:
        print(f"ERROR: 无法启动服务：{e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
