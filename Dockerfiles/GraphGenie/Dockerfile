FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install wget tar ca-certificates curl python3 python3-pip -y
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt install git-lfs -y


RUN pip3 install --no-cache-dir --break-system-packages neo4j redisgraph psycopg2-binary

WORKDIR /
RUN mkdir /code
WORKDIR /code
RUN git clone https://github.com/YuanchengJiang/GraphGenie.git

COPY ["helper-tools/run_graph_genie.sh", "/code/GraphGenie/"]
RUN chmod +x /code/GraphGenie/run_graph_genie.sh

WORKDIR /code/GraphGenie

ENTRYPOINT ["/code/GraphGenie/run_graph_genie.sh"]
