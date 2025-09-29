#!/bin/bash

set -euo pipefail

# Defaults
GRAPH_DB_TYPE="neo4j"
GRAPH_DB_VERSION="5.1.0"
GRAPH_DB_PORT="7687"
GRAPH_DB_USERNAME="neo4j"
GRAPH_DB_PASSWORD="mcgill123!"

usage() {
  cat <<'EOF'
Usage: setup_neo4j_seed.sh --graph-db-type <type> \
                           --graph-db-version <version> \
                           --graph-db-port <port> \
                           --graph-db-username <user> \
                           --graph-db-password <pass>

Options:
  --graph-db-type <type>         (optional, default: neo4j)
  --graph-db-version <version>   (optional, default: 5.1.0)
  --graph-db-port <port>         (optional, default: 7687)
  --graph-db-username <user>     (optional, default: neo4j)
  --graph-db-password <pass>     (optional, default: mcgill123!)
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
    --graph-db-type)
      [[ $# -ge 2 ]] || { echo "Missing value for --graph-db-type" >&2; usage; exit 2; }
      GRAPH_DB_TYPE="$2"; shift 2;;
    --graph-db-version)
      [[ $# -ge 2 ]] || { echo "Missing value for --graph-db-version" >&2; usage; exit 2; }
      GRAPH_DB_VERSION="$2"; shift 2;;
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
if ! [[ "$GRAPH_DB_PORT" =~ ^[0-9]+$ ]] || (( GRAPH_DB_PORT < 1 || GRAPH_DB_PORT > 65535 )); then
  echo "Error: --graph-db-port must be an integer in 1–65535 (got '$GRAPH_DB_PORT')." >&2
  exit 2
fi

# Safe summary which does NOT print the password
echo "Setting up ${GRAPH_DB_TYPE} version ${GRAPH_DB_VERSION}: localhost:${GRAPH_DB_PORT} with user ${GRAPH_DB_USERNAME} (password length=${#GRAPH_DB_PASSWORD})"

# Escape for sed replacement (escape & and \)
esc() { printf '%s' "$1" | sed -e 's/[&\\]/\\&/g'; }
type_esc=$(esc "$GRAPH_DB_TYPE")
vers_esc=$(esc "$GRAPH_DB_VERSION")
port_esc=$(esc "$GRAPH_DB_PORT")
user_esc=$(esc "$GRAPH_DB_USERNAME")
pass_esc=$(esc "$GRAPH_DB_PASSWORD")

export WORKDIR="$PWD"

if [ "$type_esc" == "neo4j" ]; then

    if [ "$vers_esc" == "5.1.0" ]; then
        # Load Neo4j
        docker pull neo4j:5.1.0-community
        docker pull neo4j/neo4j-admin:5.1.0-community

        # Create local dirs for the neo4k database
        backups_path="$WORKDIR/neo4j/versions/5.1.0/backups"
        mkdir -p -- "$backups_path"
        data_path="$WORKDIR/neo4j/versions/5.1.0/data"
        mkdir -p -- "$data_path"
        logs_path="$WORKDIR/neo4j/versions/5.1.0/logs"
        mkdir -p -- "$logs_path"

        # Load Neo4j with the Movie Reccomendation Dataset and Run the DB
        docker run --interactive --tty --rm --volume "$data_path:/data" --volume "$backups_path:/backups" neo4j/neo4j-admin:5.1.0-community neo4j-admin database load --overwrite-destination=true neo4j --from-path=/backups
        docker run --restart always --publish=7474:7474 --publish "$port_esc:7687" --env "NEO4J_AUTH=$user_esc/$pass_esc" --name "neo4j" -d --volume "$data_path:/data" --volume "$logs_path:/logs" neo4j:5.1.0-community

    elif [ "$vers_esc" == "5.4.0" ]; then
        # Load Neo4j
        docker pull neo4j:5.4.0-community
        docker pull neo4j/neo4j-admin:5.4.0-community

        # Create local dirs for the neo4k database
        backups_path="$WORKDIR/neo4j/versions/5.4.0/backups"
        mkdir -p -- "$backups_path"
        data_path="$WORKDIR/neo4j/versions/5.4.0/data"
        mkdir -p -- "$data_path"
        logs_path="$WORKDIR/neo4j/versions/5.4.0/logs"
        mkdir -p -- "$logs_path"

        # Load Neo4j with the Movie Reccomendation Dataset and Run the DB
        docker run --interactive --tty --rm --volume "$data_path:/data" --volume "$backups_path:/backups" neo4j/neo4j-admin:5.4.0-community neo4j-admin database load --overwrite-destination=true neo4j --from-path=/backups
        docker run --restart always --publish=7474:7474 --publish "$port_esc:7687" --env "NEO4J_AUTH=$user_esc/$pass_esc" --name "neo4j" -d --volume "$data_path:/data" --volume "$logs_path:/logs" neo4j:5.4.0-community
    else
        echo "neo4j version $vers_esc not currently supported"
    fi
else
    echo "$type_esc GraphDB not currently supported"
fi