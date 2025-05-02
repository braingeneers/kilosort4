FROM quay.io/ucsc_cgl/nrp:12.4.1cudnn-runtime-ubuntu22.04

RUN pip install --upgrade pip

RUN pip install git+https://github.com/OjasPBrahme/braingeneerspy.git@045bfb63cbcdab70c341e216fb6864976045ee57

RUN pip install --no-cache-dir -v git+https://github.com/braingeneers/SpikeData@da697f56860c6d9a8f60e2fb745e8841b535f9d8

RUN pip install --upgrade neo spikeinterface spikeinterface[full]


COPY run.py /tmp/run.py

