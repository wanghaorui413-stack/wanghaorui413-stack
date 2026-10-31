import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from chlorophyll.common import ROOT
from chlorophyll.inference import Predictor
from chlorophyll.serve import make_handler


@unittest.skipUnless((ROOT / "runs" / "gloria_v1" / "model_card.json").exists(), "train gloria_v1 first")
class LocalAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor("gloria_v1")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.predictor))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def request(self, method, path, payload=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=20)
        body = None if payload is None else json.dumps(payload).encode()
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        answer = response.status, json.loads(response.read().decode("utf-8"))
        connection.close()
        return answer

    def test_batch_prediction_and_model_switch(self):
        status, example = self.request("GET", "/api/example")
        self.assertEqual(status, 200)
        for model in ["selected", "mlp"]:
            status, result = self.request("POST", "/api/predict", {"csv": example["csv"], "model": model})
            self.assertEqual(status, 200)
            expected = self.predictor.from_csv(example["csv"], model)
            self.assertEqual(len(result["rows"]), 5)
            self.assertAlmostEqual(result["rows"][0]["chlorophyll_ug_l"], expected.iloc[0].chlorophyll_ug_l, places=6)
            self.assertEqual(len(result["spectrum"]["rrs"]), 71)
            self.assertIn("sample_id", result["csv"])

    def test_invalid_input_returns_400(self):
        for payload in [{"csv": "bad input"}, {"csv": 123}, {"missing": "csv"}]:
            status, result = self.request("POST", "/api/predict", payload)
            self.assertEqual(status, 400)
            self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
