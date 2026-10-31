import json
import logging
import socket
import threading
import time
import uuid
import webbrowser
from importlib.resources import files
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from .common import ROOT, WAVELENGTHS, named_path, read_json
from .data import read_prediction_csv
from .inference import Predictor

LOGGER = logging.getLogger("chlorophyll.http")
MAX_BODY_BYTES = 2_000_000
READ_TIMEOUT_SECONDS = 10


def public_status(card):
    """Explicit public DTO: never serialize source paths or internal configuration."""
    quality = card.get("dataset_quality", {})
    source = card.get("source", {})
    return {
        "schema_version": card.get("schema_version"),
        "selected_model": card["selected_model"],
        "dataset": card.get("dataset"),
        "source": {k: source[k] for k in ["name", "doi", "license"] if k in source},
        "dataset_quality": {k: quality[k] for k in ["retained_samples", "splits"] if k in quality},
        "validation_metrics": card.get("validation_metrics", {}),
    }


class BoundedHTTPServer(ThreadingHTTPServer):
    """Reject excess connections before creating a worker thread."""

    daemon_threads = True

    def __init__(self, address, handler, max_workers=4):
        self._workers = threading.BoundedSemaphore(max_workers)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._workers.acquire(blocking=False):
            request_id = uuid.uuid4().hex
            body = json.dumps({"error": "Service busy", "code": "service_busy", "request_id": request_id}).encode()
            try:
                request.settimeout(1)
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 1\r\nConnection: close\r\n"
                    + f"X-Request-ID: {request_id}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._workers.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._workers.release()


def make_handler(predictor, read_timeout=READ_TIMEOUT_SECONDS, max_inflight=2):
    inference_slots = threading.BoundedSemaphore(max_inflight)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(read_timeout)

        def log_message(self, *args):
            # Application logs below are JSON; do not log CSV bodies or raw paths.
            pass

        def begin(self):
            self.request_id = uuid.uuid4().hex
            self.started = time.monotonic()

        def send(self, status, value, content_type="application/json; charset=utf-8"):
            if not hasattr(self, "request_id"):
                self.begin()
            if isinstance(value, dict):
                value = dict(value, request_id=self.request_id)
            body = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-ID", self.request_id)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            finally:
                LOGGER.info(
                    json.dumps(
                        {
                            "event": "http_request",
                            "request_id": self.request_id,
                            "method": self.command,
                            "route": self.path.split("?")[0]
                            if self.path.split("?")[0]
                            in ["/", "/report", "/health", "/api/status", "/api/example", "/api/predict"]
                            else "unknown",
                            "status": status,
                            "duration_ms": round((time.monotonic() - self.started) * 1000, 2),
                        }
                    )
                )

        def error(self, status, code, message):
            self.send(status, {"error": message, "code": code})

        def do_GET(self):
            self.begin()
            try:
                if self.path == "/":
                    self.send(
                        200,
                        files("chlorophyll").joinpath("assets/chlorophyll.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                elif self.path == "/report" and (predictor.run / "report.html").exists():
                    self.send(200, (predictor.run / "report.html").read_bytes(), "text/html; charset=utf-8")
                elif (
                    self.path == "/satellite-report"
                    and (ROOT / "predictions/hanriver_s2b_20240531/report.html").exists()
                ):
                    self.send(
                        200,
                        (ROOT / "predictions/hanriver_s2b_20240531/report.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                elif self.path == "/health":
                    self.send(
                        200,
                        {"status": "ok", "project": "chlorophyll", "selected_model": predictor.card["selected_model"]},
                    )
                elif self.path == "/api/status":
                    result = {"model_card": public_status(predictor.card)}
                    if (predictor.run / "test_metrics.json").exists():
                        test = read_json(predictor.run / "test_metrics.json")
                        result["test"] = {
                            k: test[k] for k in ["samples", "waterbodies", "selected_model", "metrics"] if k in test
                        }
                    self.send(200, result)
                elif self.path == "/api/example":
                    dataset = predictor.card.get("dataset")
                    example = named_path("data/prepared", dataset) / "prediction_example.csv" if dataset else None
                    if example is None or not example.exists():
                        self.error(404, "example_unavailable", "独立模型包未附带训练数据示例，请导入自己的光谱CSV")
                    else:
                        self.send(200, {"csv": example.read_text(encoding="utf-8")})
                else:
                    self.error(404, "not_found", "Not found")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                LOGGER.error(
                    json.dumps({"event": "request_failed", "request_id": self.request_id, "code": "internal_error"})
                )
                self.error(500, "internal_error", "服务内部错误，请使用请求ID定位日志")

        def do_POST(self):
            self.begin()
            acquired = False
            try:
                if self.path != "/api/predict":
                    self.error(404, "not_found", "Not found")
                    return
                content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if content_type != "application/json":
                    self.error(415, "unsupported_media_type", "Content-Type must be application/json")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    self.error(413, "body_size_limit", "请求大小须为1字节到2MB")
                    return
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("请求体不完整")
                payload = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(payload, dict) or not isinstance(payload.get("csv"), str):
                    raise ValueError("JSON须包含csv文本字符串")
                model = payload.get("model", "selected")
                if not isinstance(model, str) or model not in [
                    "selected",
                    "cnn",
                    "mlp",
                    "ridge",
                    "pls",
                    "random_forest",
                    "mean_log",
                ]:
                    raise ValueError("未知模型")
                acquired = inference_slots.acquire(blocking=False)
                if not acquired:
                    self.error(503, "service_busy", "预测队列繁忙，请稍后重试")
                    return
                result = predictor.from_csv(payload["csv"], model)
                _, spectra = read_prediction_csv(payload["csv"])
                self.send(
                    200,
                    {
                        "rows": json.loads(result.to_json(orient="records")),
                        "csv": result.to_csv(index=False),
                        "unit": "ug/L",
                        "spectrum": {"wavelengths_nm": WAVELENGTHS, "rrs": spectra[0].tolist()},
                        "note": "这是原位Rrs浓度反演，不是卫星影像处理。示例是保留测试水体中的真实光谱。",
                    },
                )
            except (socket.timeout, TimeoutError):
                self.error(408, "read_timeout", "读取请求超时")
            except (ValueError, KeyError, TypeError, UnicodeError) as error:
                self.error(400, "invalid_input", str(error))
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                LOGGER.error(
                    json.dumps({"event": "request_failed", "request_id": self.request_id, "code": "inference_failed"})
                )
                self.error(500, "internal_error", "预测失败，请使用请求ID定位日志")
            finally:
                if acquired:
                    inference_slots.release()

    return Handler


def serve(run="gloria_v1", port=8766, open_browser=False, package_path=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(2)
    predictor = Predictor(run, package_path=package_path)
    server = BoundedHTTPServer(("127.0.0.1", port), make_handler(predictor))
    print(f"Chlorophyll: http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
