FROM condaforge/miniforge3:24.11.3-0

WORKDIR /workspace/SPOT-RNA

COPY environment.cpu.yml ./environment.cpu.yml

RUN conda env create -f environment.cpu.yml && conda clean -afy

ENV PATH=/opt/conda/envs/spotrna/bin:$PATH

COPY . .

RUN curl -L "https://www.dropbox.com/s/dsrcf460nbjqpxa/SPOT-RNA-models.tar.gz" -o "SPOT-RNA-models.tar.gz" \
    && tar -xzf "SPOT-RNA-models.tar.gz" \
    && rm "SPOT-RNA-models.tar.gz"

ENTRYPOINT ["python", "SPOT-RNA.py"]
CMD ["--input", "sample_inputs/single_seq.fasta", "--output", "outputs/", "--cpu", "4"]
