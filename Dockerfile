# paysim/Dockerfile
# Build stage
FROM maven:3.9.4-eclipse-temurin-17 AS builder
ARG REPO=https://github.com/EdgarLopezPhD/PaySim.git
ARG REPO_DIR=/opt/PaySim
WORKDIR /opt
RUN git clone --depth 1 ${REPO} ${REPO_DIR}
WORKDIR ${REPO_DIR}
RUN mvn -q -DskipTests package

# Runtime stage
FROM eclipse-temurin:17-jre-jammy
WORKDIR /opt/paysim

# copia o jar gerado (usa wildcard do stage builder)
COPY --from=builder /opt/PaySim/target/*.jar /opt/paysim/paysim.jar
COPY --from=builder /opt/PaySim/paramFiles /opt/paysim/paramFiles

VOLUME ["/data"]

# variável apontando para properties (ajuste se quiser passar outro)
ENV PAYSIM_PROPERTIES=/opt/paysim/paramFiles/PaySim.properties
# saída padrão: /data/transactions.csv
ENV PAYSIM_OUTPUT=/data/transactions.csv

# comando default: roda a simulação e grava CSV em /data
ENTRYPOINT ["sh", "-c", "java -jar /opt/paysim/paysim.jar ${PAYSIM_PROPERTIES} ${PAYSIM_OUTPUT}"]
