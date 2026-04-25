"""JSONL pose broadcast server for hand_localizer."""

import json
import socket
import threading
import time

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9876


class PoseServer:
    """Broadcast the latest robot-frame cube pose to connected TCP clients."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.clients: list[socket.socket] = []
        self.clients_lock = threading.Lock()
        self.server_socket: socket.socket | None = None
        self.running = False

    def start(self) -> None:
        """Start listening for client connections in a background daemon thread."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"Pose server listening on {self.host}:{self.port}")

    def _accept_loop(self) -> None:
        """Accept incoming clients until the server is stopped."""
        while self.running and self.server_socket is not None:
            try:
                client_socket, address = self.server_socket.accept()
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self.clients_lock:
                    self.clients.append(client_socket)
                    client_count = len(self.clients)
                print(f"Client connected: {address} (total: {client_count})")
            except socket.timeout:
                continue
            except OSError:
                break

    def client_count(self) -> int:
        """Return the number of currently connected clients."""
        with self.clients_lock:
            return len(self.clients)

    def broadcast(self, pose_dict: dict) -> None:
        """Send a JSONL pose message to every currently connected client."""
        payload = dict(pose_dict)
        payload["timestamp"] = time.time()
        data = (json.dumps(payload) + "\n").encode("utf-8")
        disconnected: list[socket.socket] = []
        with self.clients_lock:
            for client_socket in self.clients:
                try:
                    client_socket.sendall(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    disconnected.append(client_socket)
            for client_socket in disconnected:
                self.clients.remove(client_socket)
                try:
                    client_socket.close()
                except OSError:
                    pass
                print(f"Client disconnected (total: {len(self.clients)})")

    def stop(self) -> None:
        """Close all sockets and stop the accept loop."""
        self.running = False
        with self.clients_lock:
            for client_socket in self.clients:
                try:
                    client_socket.close()
                except OSError:
                    pass
            self.clients.clear()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
            self.server_socket = None
        print("Pose server stopped.")
