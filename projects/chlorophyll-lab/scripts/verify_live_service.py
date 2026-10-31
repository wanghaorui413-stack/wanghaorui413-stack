import http.client
import json


def request(method, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", 8767, timeout=20)
    body = None if payload is None else json.dumps(payload).encode()
    connection.request(method, path, body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    data = response.read()
    assert response.status == 200, (response.status, data[:300])
    assert response.getheader("X-Request-ID")
    connection.close()
    return data


example = json.loads(request("GET", "/api/example"))
result = json.loads(request("POST", "/api/predict", {"csv": example["csv"], "model": "selected"}))
assert len(result["rows"]) == 5
assert result["rows"][0]["model_instance_sha256"]
report = request("GET", "/satellite-report").decode("utf-8")
assert "124 / 35055" in report
print("Live engineering service: five real predictions + satellite report + request IDs PASS")
