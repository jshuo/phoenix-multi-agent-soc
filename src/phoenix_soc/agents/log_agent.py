# log_agent.py
import asyncio, json, re, signal, time
from dataclasses import dataclass
from typing import AsyncIterator, Optional, List, Dict, Any

# Optional: enable uvloop on Linux for performance
try:
    import uvloop
    uvloop.install()
except Exception:
    pass

# ---------- Utilities ----------
RFC3339 = "%Y-%m-%dT%H:%M:%S.%fZ"

def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"

# ---------- Parser / Enricher ----------
class Parser:
    def __init__(self, json_fields: Optional[List[str]] = None, regex: Optional[str] = None):
        self.json_fields = set(json_fields or [])
        self.regex = re.compile(regex) if regex else None

    def parse(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        # Try direct JSON
        if raw.startswith("{") and raw.endswith("}"):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        # Try "msg: {...json...}" embedded
        if self.json_fields:
            for fld in self.json_fields:
                m = re.search(rf'{re.escape(fld)}=(\{{.*\}})', raw)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        parsed["_raw"] = raw
                        return parsed
                    except json.JSONDecodeError:
                        continue
        # Fallback regex (e.g., "LEVEL=INFO key=a val=b")
        if self.regex:
            m = self.regex.search(raw)
            if m:
                data = m.groupdict()
                data["_raw"] = raw
                return data
        # Last resort
        return {"message": raw}

class Enricher:
    def __init__(self, static_tags: Dict[str, str] = None):
        self.static_tags = static_tags or {}

    def enrich(self, doc: Dict[str, Any], host: Optional[str]) -> Dict[str, Any]:
        doc.setdefault("@timestamp", now_ts())
        if host:
            doc.setdefault("host", host)
        for k, v in self.static_tags.items():
            doc.setdefault(k, v)
        return doc

# ---------- Sinks ----------
class PrintSink:
    def __init__(self): pass
    async def write_batch(self, batch: List[Dict[str, Any]]):
        for d in batch:
            print(json.dumps(d, ensure_ascii=False))

class OpenSearchSink:
    def __init__(self, endpoint: str, index: str, username: str = None, password: str = None, verify_certs: bool = True):
        from opensearchpy import OpenSearch
        self.index = index
        self.client = OpenSearch(
            hosts=[endpoint],
            http_auth=(username, password) if username else None,
            use_ssl=endpoint.startswith("https"),
            verify_certs=verify_certs,
            ssl_show_warn=False,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

    async def write_batch(self, batch: List[Dict[str, Any]]):
        # Use thread executor (client is sync); keep example simple.
        loop = asyncio.get_running_loop()
        def do_bulk():
            from opensearchpy.helpers import bulk
            actions = ({"_index": self.index, "_source": d} for d in batch)
            return bulk(self.client, actions)
        await loop.run_in_executor(None, do_bulk)

# ---------- Collectors ----------
class SyslogCollector:
    """
    Minimal syslog server (UDP and TCP). Configure remote hosts to forward to this agent.
    """
    def __init__(self, host="0.0.0.0", udp_port=514, tcp_port=5514):
        self.host = host
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self._queue: asyncio.Queue = asyncio.Queue()
        self._servers: List[asyncio.AbstractServer] = []
        self._udp_transport = None

    async def start(self):
        loop = asyncio.get_running_loop()

        # UDP
        class _UDP(asyncio.DatagramProtocol):
            def __init__(self, queue): self.queue = queue
            def datagram_received(self, data, addr):
                host, _ = addr
                msg = data.decode(errors="ignore")
                self.queue.put_nowait((host, msg))

        self._udp_transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDP(self._queue),
            local_addr=(self.host, self.udp_port)
        )

        # TCP
        async def handle_tcp(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            addr = writer.get_extra_info('peername')
            host = addr[0] if addr else None
            try:
                while not reader.at_eof():
                    line = await reader.readline()
                    if not line:
                        break
                    self._queue.put_nowait((host, line.decode(errors="ignore")))
            finally:
                writer.close()
                await writer.wait_closed()

        tcp_srv = await asyncio.start_server(handle_tcp, self.host, self.tcp_port)
        self._servers.append(tcp_srv)
        print(f"[SyslogCollector] UDP {self.udp_port}, TCP {self.tcp_port} started")

    async def stop(self):
        if self._udp_transport:
            self._udp_transport.close()
        for s in self._servers:
            s.close()
            await s.wait_closed()

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            host, line = await self._queue.get()
            yield {"host": host, "line": line}

class SSHTailCollector:
    """
    SSH-tail remote files (like `tail -F`). Keeps a persistent session and reconnects on failure.
    """
    def __init__(self, host: str, user: str, key_path: Optional[str], files: List[str], port: int = 22):
        self.host = host
        self.user = user
        self.key_path = key_path
        self.files = files
        self.port = port
        self._queue: asyncio.Queue = asyncio.Queue()

    async def _reader_loop(self):
        import paramiko
        while True:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                if self.key_path:
                    client.connect(self.host, port=self.port, username=self.user, key_filename=self.key_path, timeout=15)
                else:
                    # NOTE: For password, extend to read from env/secret store
                    raise RuntimeError("For security, use key auth. (Extend if you need password auth.)")

                # Use tail -F across files; could also use journald's journalctl -f
                cmd = "tail -n0 -F " + " ".join(self.files)
                transport = client.get_transport()
                channel = transport.open_session()
                channel.exec_command(cmd)

                buf = b""
                while True:
                    if channel.recv_ready():
                        data = channel.recv(65536)
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._queue.put_nowait((self.host, line.decode(errors="ignore")))
                    else:
                        await asyncio.sleep(0.05)

            except Exception as e:
                print(f"[SSHTailCollector] {self.host} error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    async def start(self):
        asyncio.create_task(self._reader_loop())
        print(f"[SSHTailCollector] started for {self.host}:{self.port} -> {self.files}")

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        while True:
            host, line = await self._queue.get()
            yield {"host": host, "line": line}

# ---------- Pipeline Runner ----------
@dataclass
class PipelineConfig:
    batch_size: int = 500
    flush_seconds: float = 2.0
    max_queue: int = 50_000

class Pipeline:
    def __init__(self, parser: Parser, enricher: Enricher, sink, config: PipelineConfig):
        self.parser = parser
        self.enricher = enricher
        self.sink = sink
        self.cfg = config
        self.q: asyncio.Queue = asyncio.Queue(self.cfg.max_queue)
        self._stop = asyncio.Event()

    async def put_raw(self, host: str, raw: str):
        # Ingest path with backpressure
        await self.q.put((host, raw))

    async def _flusher(self):
        batch: List[Dict[str, Any]] = []
        last = time.monotonic()
        while not self._stop.is_set():
            timeout = self.cfg.flush_seconds
            try:
                host, raw = await asyncio.wait_for(self.q.get(), timeout=timeout)
                doc = self.parser.parse(raw)
                doc = self.enricher.enrich(doc, host)
                batch.append(doc)
                if len(batch) >= self.cfg.batch_size:
                    await self._flush(batch)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self._flush(batch)
                    batch = []
                continue
        # drain
        if batch:
            await self._flush(batch)

    async def _flush(self, batch: List[Dict[str, Any]]):
        for _ in range(3):
            try:
                await self.sink.write_batch(batch)
                return
            except Exception as e:
                print(f"[Pipeline] flush error: {e}; retrying in 1s...")
                await asyncio.sleep(1)
        print("[Pipeline] dropped batch after retries")

    async def start(self, collectors: List[Any]):
        # Start collectors and multiplex their streams to the pipeline
        for c in collectors:
            await getattr(c, "start")()
        asyncio.create_task(self._flusher())

        async def pump(collector):
            async for item in collector.stream():
                await self.put_raw(item.get("host"), item.get("line"))

        for c in collectors:
            asyncio.create_task(pump(c))

    async def stop(self):
        self._stop.set()

# ---------- Bring it together ----------
async def main():
    # Choose ONE sink to start:
    sink = PrintSink()
    # Example OpenSearch sink:
    # sink = OpenSearchSink(endpoint="https://localhost:9200", index="logs-agent-*", username="admin", password="admin", verify_certs=False)

    parser = Parser(
        json_fields=["msg", "message_json"],      # if logs embed JSON in a field
        regex=r"(?P<level>INFO|WARN|ERROR)\s+(?P<kv>.*)"  # example fallback
    )
    enricher = Enricher(static_tags={"app": "log-agent", "env": "dev"})
    pipeline = Pipeline(parser, enricher, sink, PipelineConfig(batch_size=200, flush_seconds=1.0))

    # Collectors:
    syslog = SyslogCollector(host="0.0.0.0", udp_port=5514, tcp_port=5515)  # non-privileged ports
    # Or SSH tail from remote:
    ssh = SSHTailCollector(host="10.0.0.25", user="ubuntu", key_path="~/.ssh/id_rsa",
                           files=["/var/log/syslog", "/var/log/auth.log"], port=22)

    await pipeline.start([syslog, ssh])

    # Graceful shutdown
    stop_ev = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop_ev.set)
        except NotImplementedError:
            pass
    print("[Main] log agent running. Press Ctrl+C to stop.")
    await stop_ev.wait()
    print("[Main] stopping...")
    await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())
