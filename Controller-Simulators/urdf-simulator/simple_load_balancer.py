#!/usr/bin/env python3
"""
Simple Round-Robin Load Balancer for PyBullet Simulator Instances

Always listens on port 5000 and balances across specified backend ports
"""

import argparse
import http.server
import socketserver
import urllib.request
import urllib.parse
import threading
from typing import List
import sys

class LoadBalancingHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, backends: List[int], *args, **kwargs):
        self.backends = backends
        self.current_backend = 0
        self.lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def get_next_backend(self) -> int:
        with self.lock:
            backend = self.backends[self.current_backend]
            self.current_backend = (self.current_backend + 1) % len(self.backends)
            return backend

    def forward_request(self, backend_port: int):
        # Reconstruct the original request URL for the backend
        path = self.path
        if '?' in path:
            path, query = path.split('?', 1)
            query = '?' + query
        else:
            query = ''

        backend_url = f"http://localhost:{backend_port}{path}{query}"

        # Forward the request method
        method = self.command

        # Prepare headers (excluding Host and Connection)
        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ['host', 'connection']
        }

        # Create the request
        req = urllib.request.Request(
            backend_url,
            method=method,
            headers=headers,
            data=self.rfile.read(int(self.headers.get('Content-Length', 0))) if method in ['POST', 'PUT', 'PATCH'] else None
        )

        try:
            with urllib.request.urlopen(req) as response:
                # Copy response status
                self.send_response(response.status)

                # Copy response headers
                for header, value in response.getheaders():
                    self.send_header(header, value)
                self.end_headers()

                # Stream response body
                self.wfile.write(response.read())
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")

    def do_GET(self):
        backend_port = self.get_next_backend()
        print(f"Forwarding GET {self.path} to port {backend_port}")
        self.forward_request(backend_port)

    def do_POST(self):
        backend_port = self.get_next_backend()
        print(f"Forwarding POST {self.path} to port {backend_port}")
        self.forward_request(backend_port)

    def do_PUT(self):
        backend_port = self.get_next_backend()
        print(f"Forwarding PUT {self.path} to port {backend_port}")
        self.forward_request(backend_port)

    def do_PATCH(self):
        backend_port = self.get_next_backend()
        print(f"Forwarding PATCH {self.path} to port {backend_port}")
        self.forward_request(backend_port)

    def do_DELETE(self):
        backend_port = self.get_next_backend()
        print(f"Forwarding DELETE {self.path} to port {backend_port}")
        self.forward_request(backend_port)

def parse_backend_ports(start_port: int, count: int) -> List[int]:
    """Generate list of backend ports starting from start_port+1"""
    # Backend instances should be on ports 5001 to 5001+count-1 when start_port is 5000
    return list(range(5001, 5001 + count))

def run_server(backends: List[int]):
    handler_class = lambda *args, **kwargs: LoadBalancingHandler(backends, *args, **kwargs)
    with socketserver.TCPServer(("", 5000), handler_class) as httpd:
        print(f"Load balancer running on port 5000")
        print(f"Balancing across backends: {', '.join(map(str, backends))}")
        print("Press Ctrl+C to stop...")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down load balancer...")
            httpd.shutdown()

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_load_balancer.py <count>")
        print("Example: python simple_load_balancer.py 10")
        sys.exit(1)

    try:
        count = int(sys.argv[1])
        backends = parse_backend_ports(count)

        if count < 1:
            raise ValueError("Count must be at least 1")

        run_server(backends)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()