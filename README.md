# GDSynth
A Graph Database Testing Framework via Effective Graph Synthesis

## Requirements
Python 3.11
Pipenv
Docker

## Running Instructions
1. pipenv install
2. pipenv shell
1. ./helper-tools/setup_neo4j_seed.sh
2. python3 gd_synth.py --database_ip 0.0.0.0 --database_port 7687 --database_username neo4j --database_password mcgill123! --search_space_file Input/search_space_definition_mini_sample.json --log_level info --max_duration 360