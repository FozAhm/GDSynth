#!/bin/bash
set -euo pipefail

INI_PATH="/code/GraphGenie/graphgenie.ini"

GRAPH_DB_IP=""
GRAPH_DB_PORT="7687"         # default
GRAPH_DB_USERNAME=""
GRAPH_DB_PASSWORD=""

usage() {
  cat <<'EOF'
Usage: run_graph_genie.sh --graph-db-ip <ip_or_host> \
                          --graph-db-port <port> \
                          --graph-db-username <user> \
                          --graph-db-password <pass>

Options:
  --graph-db-ip <ip_or_host>     (required)
  --graph-db-port <port>         (optional, default: 7687)
  --graph-db-username <user>     (required)
  --graph-db-password <pass>     (required)
  -h, --help                     Show this help and exit

Notes:
  * Password is accepted ONLY via CLI option as requested. No env vars used.
  * Be aware: CLI args are visible to local process listings (ps/top).
EOF
}

# Long-option parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage; exit 0;;
    --graph-db-ip)
      [[ $# -ge 2 ]] || { echo "Missing value for --graph-db-ip" >&2; usage; exit 2; }
      GRAPH_DB_IP="$2"; shift 2;;
    --graph-db-port)
      [[ $# -ge 2 ]] || { echo "Missing value for --graph-db-port" >&2; usage; exit 2; }
      GRAPH_DB_PORT="$2"; shift 2;;
    --graph-db-username)
      [[ $# -ge 2 ]] || { echo "Missing value for --graph-db-username" >&2; usage; exit 2; }
      GRAPH_DB_USERNAME="$2"; shift 2;;
    --graph-db-password)
      [[ $# -ge 2 ]] || { echo "Missing value for --graph-db-password" >&2; usage; exit 2; }
      GRAPH_DB_PASSWORD="$2"; shift 2;;
    --) shift; break;;   # end of options
    -*)
      echo "Unknown option: $1" >&2; usage; exit 2;;
    *)
      # positional args (ignored here but can be collected if needed)
      shift;;
  esac
done

# Validation
[[ -n "$GRAPH_DB_IP" ]] || { echo "Error: --graph-db-ip is required." >&2; exit 2; }
[[ -n "$GRAPH_DB_USERNAME" ]] || { echo "Error: --graph-db-username is required." >&2; exit 2; }
[[ -n "$GRAPH_DB_PASSWORD" ]] || { echo "Error: --graph-db-password is required." >&2; exit 2; }
if ! [[ "$GRAPH_DB_PORT" =~ ^[0-9]+$ ]] || (( GRAPH_DB_PORT < 1 || GRAPH_DB_PORT > 65535 )); then
  echo "Error: --graph-db-port must be an integer in 1–65535 (got '$GRAPH_DB_PORT')." >&2
  exit 2
fi

# Safe summary which does NOT print the password
echo "Graph DB: ${GRAPH_DB_IP}:${GRAPH_DB_PORT} with user ${GRAPH_DB_USERNAME} (password length=${#GRAPH_DB_PASSWORD})"

# Escape for sed replacement (escape & and \)
esc() { printf '%s' "$1" | sed -e 's/[&\\]/\\&/g'; }
ip_esc=$(esc "$GRAPH_DB_IP")
port_esc=$(esc "$GRAPH_DB_PORT")
user_esc=$(esc "$GRAPH_DB_USERNAME")
pass_esc=$(esc "$GRAPH_DB_PASSWORD")

# Update only the [default] section
# This replaces existing keys in the default; section
# A backup "graph genie.ini.bak" is also created automatically
sed -E -i.bak \
  -e "/^\[default\]/, /^\[/{ 
        s|^[[:space:]]*graphdb[[:space:]]*=.*|graphdb = neo4j|;
        s|^[[:space:]]*ip[[:space:]]*=.*|ip = ${ip_esc}|;
        s|^[[:space:]]*port[[:space:]]*=.*|port = ${port_esc}|;
        s|^[[:space:]]*username[[:space:]]*=.*|username = ${user_esc}|;
        s|^[[:space:]]*password[[:space:]]*=.*|password = ${pass_esc}|;
     }" "$INI_PATH"

echo "Updated [$INI_PATH] [default]: ip=${GRAPH_DB_IP}, port=${GRAPH_DB_PORT}, username=${GRAPH_DB_USERNAME}, password=(redacted)"

python3 /code/GraphGenie/main.py

