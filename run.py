"""
Application Entrypoint
Run with:
  flask --app app run --debug
or:
  python run.py
"""

import os
import socket
from app import create_app

app = create_app()

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

if __name__ == '__main__':
    default_port = 5001 if is_port_in_use(5000) else 5000
    port = int(os.environ.get('PORT', default_port))
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

