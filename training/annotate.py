"""Local annotation tool (stdlib only, single HTML page, binds to 127.0.0.1).

Listen to each clip, correct the transcript, verify the speaker, set emotion/style/role, or reject noisy /
misaligned clips. Saving a clip with any label sets label_source=human_verified (plus verified_by/verified_at);
nothing is ever pre-filled or guessed. Writes back to the manifest JSONL atomically.

  python -m training.annotate --manifest data/manifest.jsonl --annotator NAME [--port 8765]
"""
import argparse
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from training.manifest import HUMAN, LABEL_FIELDS, read_rows, write_jsonl

EDITABLE = ("text", "speaker_id", *LABEL_FIELDS)
_LOCK = threading.Lock()

PAGE = """<!doctype html><meta charset=utf-8><title>Annotate</title>
<style>body{font:15px sans-serif;max-width:720px;margin:2em auto;padding:0 1em}input,textarea{width:100%;box-sizing:border-box;margin:.2em 0 .8em}
button{margin-right:.5em;padding:.4em .9em}#m{color:#555}</style>
<h2>Clip <span id=n></span></h2><div id=m></div><audio id=a controls style="width:100%"></audio>
<label>Transcript</label><textarea id=text rows=3></textarea>
<label>Speaker ID (verify by listening)</label><input id=speaker_id>
<label>Emotion (leave blank if unsure)</label><input id=emotion>
<label>Style</label><input id=style><label>Role</label><input id=role>
<button onclick=save(0)>Save &amp; verify</button><button onclick=save(1)>Reject (noisy/misaligned)</button>
<button onclick=go(-1)>Prev</button><button onclick=go(1)>Next</button>
<script>
let items=[],i=0;const $=id=>document.getElementById(id);
async function load(){items=await (await fetch('/api/items')).json();i=Math.max(0,items.findIndex(x=>!x.label_source&&!x.rejected));show()}
function show(){const x=items[i];if(!x)return;$('n').textContent=(i+1)+'/'+items.length;
$('m').textContent='id '+x.id+' | lang '+x.language+' | label_source '+(x.label_source||'none')+(x.rejected?' | REJECTED '+(x.reject_reason||''):'');
$('a').src='/audio/'+x.id;for(const k of ['text','speaker_id','emotion','style','role'])$(k).value=x[k]||''}
function go(d){i=Math.min(items.length-1,Math.max(0,i+d));show()}
async function save(rej){const x=items[i];const b={id:x.id,rejected:!!rej};
for(const k of ['text','speaker_id','emotion','style','role'])b[k]=$(k).value;
if(rej)b.reject_reason=prompt('Reason (noisy / misaligned / other)')||'rejected';
const r=await fetch('/api/update',{method:'POST',body:JSON.stringify(b)});items[i]=await r.json();show()}
load()
</script>"""


def apply_update(rows: list[dict], upd: dict, annotator: str) -> dict:
    """Apply one human edit to the row with upd['id'] (in place) and return it. Labels => human_verified."""
    row = next((r for r in rows if r["id"] == upd["id"]), None)
    if row is None:
        raise KeyError(upd["id"])
    for k in EDITABLE:
        if k in upd:
            v = upd[k].strip() if isinstance(upd[k], str) else upd[k]
            row[k] = v or None if k != "text" else (v if v else row.get("text"))
    if upd.get("rejected"):
        row["rejected"] = True
        row["reject_reason"] = upd.get("reject_reason") or "rejected"
    else:
        row["rejected"] = False
        row["reject_reason"] = None
        row["label_source"] = HUMAN
        row["verified_by"] = annotator
        row["verified_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def make_handler(manifest: Path, annotator: str):
    base = manifest.parent

    def audio_path(row):
        p = Path(row["audio"])
        return p if p.is_absolute() else base / p

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            with _LOCK:
                rows = read_rows(manifest)
            if self.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/items":
                return self._send(200, json.dumps(rows, ensure_ascii=False).encode())
            if self.path.startswith("/audio/"):  # only ids present in the manifest; never a raw filesystem path
                row = next((r for r in rows if r["id"] == self.path[7:]), None)
                if row and audio_path(row).is_file():
                    return self._send(200, audio_path(row).read_bytes(), "audio/wav")
            self._send(404, b"{}")

        def do_POST(self):
            if self.path != "/api/update":
                return self._send(404, b"{}")
            upd = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with _LOCK:
                rows = read_rows(manifest)
                try:
                    row = apply_update(rows, upd, annotator)
                except KeyError:
                    return self._send(404, b"{}")
                write_jsonl(manifest, rows)
            self._send(200, json.dumps(row, ensure_ascii=False).encode())

    return H


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="JSONL manifest (convert CSV first: training.manifest)")
    p.add_argument("--annotator", required=True, help="your name; stored as verified_by")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if a.manifest.suffix != ".jsonl":
        raise SystemExit("annotation tool edits a .jsonl manifest")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.manifest, a.annotator))
    print(f"open http://127.0.0.1:{a.port}/  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
