FROM tensorflow/tensorflow:1.15.5-py3

WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "SPOT-RNA.py"]
CMD ["--help"]
