sudo -u azureuser bash -lc '
set -euo pipefail
PKG_DIR="/home/azureuser/qwen-env/lib/python3.10/site-packages/pyairports"
mkdir -p "$PKG_DIR"
cat <<'PY' > "$PKG_DIR/__init__.py"
from .airports import AIRPORT_LIST
PY
cat <<'PY' > "$PKG_DIR/airports.py"
AIRPORT_LIST = []
PY
'
