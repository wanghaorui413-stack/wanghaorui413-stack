"""Service tests independent of trained artifacts and external data."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import shutil
from chlorophyll.serve import BoundedHTTPServer, make_handler


class BrowserStateTests(unittest.TestCase):
    def test_result_export_is_invalidated_on_change_failure_and_stale_response(self):
        node = shutil.which("node")
        if not node:
            bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
            node = str(bundled) if bundled.exists() else None
        if not node:
            self.skipTest("Node is needed for the isolated JavaScript state test")
        page = Path(__file__).resolve().parents[1] / "web/chlorophyll.html"
        harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes={};
function element(){return {disabled:false,value:'data',children:[],events:{},textContent:'',
 addEventListener(k,f){this.events[k]=f},replaceChildren(){this.children=[]},
 append(x){this.children.push(x)},setAttribute(){},click(){}}}
const ctx={document:{getElementById(k){return nodes[k]??=(element())},createElement:element,createElementNS:element},
 fetch:()=>new Promise(()=>{}),Blob:class{},URL:{createObjectURL(){return 'blob:test'},revokeObjectURL(){}}};
vm.createContext(ctx);vm.runInContext(fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0],ctx);
const data={rows:[{sample_id:'a',chlorophyll_ug_l:2,selected_model:'cnn',max_abs_z:1}],csv:'export',spectrum:{rrs:Array(71).fill(.01)}};
const response={ok:true,json:async()=>data};
(async()=>{
 ctx.fetch=async()=>response; await nodes.predict.onclick();assert.equal(nodes.download.disabled,false);
 nodes.csv.events.input();assert.equal(nodes.download.disabled,true);assert.equal(nodes.results.children.length,0);
 await nodes.predict.onclick();ctx.fetch=async()=>{throw Error('failure')};await nodes.predict.onclick();
 assert.equal(nodes.download.disabled,true);assert.equal(vm.runInContext('outputCSV',ctx),'');
 let finish;ctx.fetch=()=>new Promise(r=>finish=r);const pending=nodes.predict.onclick();
 nodes.model.events.change();finish(response);await pending;
 assert.equal(nodes.download.disabled,true);assert.equal(nodes.results.children.length,0);
})().catch(e=>{console.error(e);process.exitCode=1});
"""
        completed = subprocess.run([node, "-e", harness, str(page)], capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class Stub:
    def __init__(self, directory):
        self.run = Path(directory)
        self.card = {
            "selected_model": "cnn",
            "source": {"name": "fixture", "input_file": "C:/private/file.csv"},
            "config": {"path": "C:/private"},
            "dataset_quality": {},
        }

    def from_csv(self, text, model):
        if text == "crash":
            raise RuntimeError("private detail")
        raise ValueError("invalid spectral input")


class ServiceEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = BoundedHTTPServer(("127.0.0.1", 0), make_handler(Stub(self.temp.name), read_timeout=0.15))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method="GET", path="/health", payload=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        client.request(
            method,
            path,
            body=None if payload is None else json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        response = client.getresponse()
        result = response.status, dict(response.getheaders()), json.loads(response.read())
        client.close()
        return result

    def test_public_status_omits_internal_paths(self):
        status, headers, body = self.request(path="/api/status")
        self.assertEqual(status, 200)
        self.assertNotIn("private", json.dumps(body))
        self.assertNotIn("config", body["model_card"])
        self.assertEqual(headers["X-Request-ID"], body["request_id"])

    def test_failure_returns_json_then_recovers(self):
        status, _, body = self.request("POST", "/api/predict", {"csv": "crash"})
        self.assertEqual(status, 500)
        self.assertEqual(body["code"], "internal_error")
        self.assertNotIn("private", json.dumps(body))
        self.assertEqual(self.request()[0], 200)
        self.assertEqual(self.request("POST", "/api/predict", {"csv": "bad"})[0], 400)
        self.assertEqual(self.request()[0], 200)

    def test_body_read_timeout_and_recovery(self):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        client.putrequest("POST", "/api/predict")
        client.putheader("Content-Type", "application/json")
        client.putheader("Content-Length", "100")
        client.endheaders()
        response = client.getresponse()
        self.assertEqual(response.status, 408)
        self.assertEqual(json.loads(response.read())["code"], "read_timeout")
        client.close()
        self.assertEqual(self.request()[0], 200)

    def test_release_without_dataset_has_stable_example_error(self):
        status, _, body = self.request(path="/api/example")
        self.assertEqual(status, 404)
        self.assertEqual(body["code"], "example_unavailable")

    def test_structured_request_log(self):
        with self.assertLogs("chlorophyll.http", level="INFO") as logs:
            _, _, body = self.request()
        event = json.loads(logs.records[-1].getMessage())
        self.assertEqual(event["request_id"], body["request_id"])
        self.assertGreaterEqual(event["duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()
