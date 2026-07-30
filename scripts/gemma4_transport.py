"""POST JSON to a llama-server over TCP or a Unix socket.

Shared by the gemma4-* live scripts. It exists because the packaged router
listens on a Unix socket and nothing else -- a single-user machine exposes
nothing to the network, so access is filesystem permissions -- and `requests`
cannot dial one without an extra dependency. `http.client` needs only its
connect() replaced.

Keep this the only copy: a second one drifts, and the first version of the
override probe already produced a confidently wrong claim from a transport-level
detail (comparing rendered token text instead of ids).
"""
import http.client
import json
import socket
import urllib.parse


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float = 600.0):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.path)
        self.sock = s


def connect(base_url: str, timeout: float = 600.0) -> http.client.HTTPConnection:
    """`unix:///path/to.sock` or `http://host:port`."""
    if base_url.startswith("unix://"):
        return UnixHTTPConnection(base_url[len("unix://"):], timeout=timeout)
    u = urllib.parse.urlparse(base_url)
    return http.client.HTTPConnection(u.hostname or "localhost", u.port or 80, timeout=timeout)


def post_json(base_url: str, route: str, body: dict,
              timeout: float = 600.0) -> tuple[int, str]:
    conn = connect(base_url, timeout)
    try:
        conn.request("POST", route, body=json.dumps(body),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", "replace")
    finally:
        conn.close()
