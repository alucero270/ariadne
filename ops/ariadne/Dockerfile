# Build from the repository root: docker build -f ops/ariadne/Dockerfile .
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --locked -p ariadne

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 ariadne \
 && mkdir -p /data && chown ariadne /data
COPY --from=build /src/target/release/ariadne /usr/local/bin/ariadne
COPY eval /app/eval
USER ariadne
ENV ARIADNE_ROOT=/source ARIADNE_DB=/data/ariadne.db ARIADNE_BIND=0.0.0.0:8090
EXPOSE 8090
ENTRYPOINT ["ariadne"]
CMD ["serve"]
