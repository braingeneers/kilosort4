FROM quay.io/ucsc_cgl/nrp:12.4.1cudnn-runtime-ubuntu22.04

RUN pip install --upgrade kilosort==4.0.39

RUN pip install --upgrade spikeinterface==0.103.0

COPY run.py /tmp/run.py
