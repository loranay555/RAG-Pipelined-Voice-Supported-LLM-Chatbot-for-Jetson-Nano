# syntax=docker/dockerfile:1.6
# llama.cpp server, CUDA build for Jetson Orin (sm_87).
ARG L4T_CUDA_DEVEL=nvcr.io/nvidia/l4t-cuda:12.6.11-devel
ARG L4T_CUDA_RUNTIME=nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

# --------------------------------------------------------------------- build
FROM ${L4T_CUDA_DEVEL} AS build

ARG LLAMA_CPP_REF=master
ARG CUDA_ARCH=87
# nvcc is memory hungry; >4 parallel jobs will OOM an 8 GB Orin during build.
ARG BUILD_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp .

# ld resolves a library's *indirect* dependencies (libggml-cuda.so carries
# DT_NEEDED libcuda.so.1) by searching -rpath-link / -rpath / LD_LIBRARY_PATH /
# system dirs -- NOT -L. And it looks for the literal SONAME, libcuda.so.1,
# while the toolkit stub is named libcuda.so. Hence this symlink; the linker
# flags below add the directory to the -rpath-link search path.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Two things worth knowing about this step:
#
# 1. The linker flags point at the CUDA driver stub. libcuda.so.1 is the real
#    driver and is injected by the nvidia container runtime only at *run* time,
#    so it is absent during `docker build`. -rpath-link (not -rpath) is what
#    actually matters here: it is a link-time-only search path, so the stub does
#    not end up in the binary's RUNPATH and the real driver is loaded on device.
#
# 2. The cache mount makes a retry resume instead of recompiling ~440 files.
#    Files written into a cache mount are NOT part of the image layer, so the
#    binaries are copied out to /out before the RUN ends.
RUN --mount=type=cache,id=llama-cpp-build,target=/src/build,sharing=locked \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DGGML_NATIVE=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release --parallel ${BUILD_JOBS} --target llama-server \
    && mkdir -p /out && cp -a build/bin/. /out/

# ------------------------------------------------------------------- runtime
FROM ${L4T_CUDA_RUNTIME} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama:${LD_LIBRARY_PATH}

COPY entrypoint.sh /opt/llama/entrypoint.sh
RUN chmod +x /opt/llama/entrypoint.sh

EXPOSE 8001
ENTRYPOINT ["/opt/llama/entrypoint.sh"]
