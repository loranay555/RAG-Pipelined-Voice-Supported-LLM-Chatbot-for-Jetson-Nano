# syntax=docker/dockerfile:1.6
# whisper.cpp server, CUDA build for Jetson Orin (sm_87).
ARG L4T_CUDA_DEVEL=nvcr.io/nvidia/l4t-cuda:12.6.11-devel
ARG L4T_CUDA_RUNTIME=nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

# --------------------------------------------------------------------- build
FROM ${L4T_CUDA_DEVEL} AS build

ARG WHISPER_CPP_REF=master
ARG CUDA_ARCH=87
ARG BUILD_JOBS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" https://github.com/ggml-org/whisper.cpp .

# ld resolves a library's *indirect* dependencies (libggml-cuda.so carries
# DT_NEEDED libcuda.so.1) by searching -rpath-link / -rpath / LD_LIBRARY_PATH /
# system dirs -- NOT -L. And it looks for the literal SONAME, libcuda.so.1,
# while the toolkit stub is named libcuda.so. Hence this symlink; the linker
# flags below add the directory to the -rpath-link search path.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# See services/llm/Dockerfile for why the CUDA driver stub is on the link line
# and why the build directory is a cache mount. Note the distinct cache id:
# sharing /src/build with the llama.cpp build would corrupt both.
RUN --mount=type=cache,id=whisper-cpp-build,target=/src/build,sharing=locked \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_SDL2=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build --config Release --parallel ${BUILD_JOBS} --target whisper-server \
    && mkdir -p /out && cp -a build/bin/. /out/

# ------------------------------------------------------------------- runtime
FROM ${L4T_CUDA_RUNTIME} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/ /opt/whisper/
ENV LD_LIBRARY_PATH=/opt/whisper:${LD_LIBRARY_PATH}

COPY entrypoint.sh /opt/whisper/entrypoint.sh
RUN chmod +x /opt/whisper/entrypoint.sh

EXPOSE 8003
ENTRYPOINT ["/opt/whisper/entrypoint.sh"]
