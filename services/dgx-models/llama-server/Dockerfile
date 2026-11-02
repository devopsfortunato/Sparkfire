# The chat LLM, Ternary Bonsai 2 27B: llama.cpp's llama-server (ggml on CUDA), which speaks the OpenAI API itself,
# so it needs no adapter. The model is a ternary rebuild of Qwen3.8-27B from prism-ml/Ternary-Bonsai-2-27B-gguf,
# and its PTQ1_0 packing needs the ternary hybrid-attention kernels in PrismML's llama.cpp fork, which upstream
# llama.cpp does not have. Like Parakeet, both stages use the PyTorch base image the other GPU services use, so
# the CUDA toolkit that builds the engine is the one that runs it, and its layers are already on the Spark.
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3@sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70
ARG LLAMA_CPP_REPO=https://github.com/PrismML-Eng/llama.cpp.git
ARG LLAMA_CPP_REF=prism
ARG LLAMA_CPP_COMMIT=1a07bfa5f4144274c8f1c9963821dd9d9a51854b

FROM ${PYTORCH_IMAGE} AS build

ARG LLAMA_CPP_REPO
ARG LLAMA_CPP_REF
ARG LLAMA_CPP_COMMIT

# The pinned base already has cmake, git and the CUDA 13 toolkit, so nothing is installed from apt.
# The commit is an older point on the fork's branch, not its tip, so the clone takes the branch's history but no
# file blobs (a partial clone: git fetches a tree's blobs when it checks it out), and the build stops unless HEAD
# is the exact commit that was pinned.
RUN git clone --filter=blob:none --branch "${LLAMA_CPP_REF}" "${LLAMA_CPP_REPO}" /src/llama.cpp \
    && git -C /src/llama.cpp checkout "${LLAMA_CPP_COMMIT}" \
    && test "$(git -C /src/llama.cpp rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}"

# Arch 121 is the GB10 (sm_121); ggml's CMake turns it into 121a itself. Static libraries keep ggml and the
# vision add-on's mtmd inside the one binary. Models come from the read-only mount, so no download support.
RUN cmake -S /src/llama.cpp -B /src/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=121 \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /src/build --target llama-server -j 8 \
    && install -D "$(find /src/build -name llama-server -type f -perm -u+x | head -n 1)" /out/llama-server \
    && ! ldd /out/llama-server | grep -Ei "libggml|libllama|libmtmd"

FROM ${PYTORCH_IMAGE}

COPY --from=build /out/llama-server /usr/local/bin/llama-server

# The server only listens on an unprivileged port and reads the read-only model mount, so it needs no root
# (uid 65534 is nobody). The check mirrors the compose health check; PORT keeps it right if the port moves.
USER 65534:65534
HEALTHCHECK --interval=30s --timeout=5s --start-period=2m --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8331}/health" || exit 1

ENTRYPOINT ["llama-server"]
