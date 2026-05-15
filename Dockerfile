FROM ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2

COPY patches/patch_modelopt_nvfp4_awq.py /tmp/patch_modelopt_nvfp4_awq.py
RUN python3 /tmp/patch_modelopt_nvfp4_awq.py && rm /tmp/patch_modelopt_nvfp4_awq.py

WORKDIR /opt/gemma31-dflash
COPY docker/entrypoint.sh /opt/gemma31-dflash/entrypoint.sh
COPY scripts/bench_categories_stream.py /opt/gemma31-dflash/scripts/bench_categories_stream.py
RUN sed -i 's/\r$//' /opt/gemma31-dflash/entrypoint.sh && \
    chmod +x /opt/gemma31-dflash/entrypoint.sh

ENTRYPOINT ["/opt/gemma31-dflash/entrypoint.sh"]
CMD ["serve"]

LABEL org.opencontainers.image.title="gemma-4-31b-uncensored-nvfp4-dflash"
LABEL org.opencontainers.image.description="DGX Spark / GB10 vLLM image for Gemma 4 31B Deckard Heretic Uncensored NVFP4 with z-lab DFlash speculative decoding."
LABEL org.opencontainers.image.source="https://github.com/AEON-7/Gemma-4-31B-Uncensored-NVFP4-DFlash"
LABEL org.opencontainers.image.documentation="https://github.com/AEON-7/Gemma-4-31B-Uncensored-NVFP4-DFlash/blob/main/README.md"
LABEL org.opencontainers.image.licenses="Apache-2.0"
