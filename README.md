# Kilosort4 Workflow

Intended to be run on the [National Research Platform (NRP)](https://nrp.ai/documentation/).  Given a Maxwell hdf5 filepath in s3, this can launch a Job on a Pod, download the Maxwell hdf5 file, run kilosort4 on it, and upload the data back to s3.

NOTE: Currently uses the default parameters that SpikeInterface chooses.

### Running on the NRP

To launch this container as a basic Job, modify the kubernetes yaml file with your Maxwell hdf5 filepath in s3, and run the example kubernetes yaml file with:

    kubectl apply -f run.yaml

For example, given a Maxwell hdf5 s3 input path at:

	"s3://braingeneers/ephys/{uuid}/original/data/{filename}.raw.hdf5"
 
 this will deposit kilosort4 results at:
 
 	"s3://braingeneers/ephys/{uuid}/derived/{filename}.raw.hdf5.ks4.zip"

### Building and Publishing the Container

To build the kilosort4 container, login to quay.io to host the docker image in quay:

	docker login quay.io

To build the image:

	docker build . -t quay.io/ucsc_cgl/kilosort4:12.4.1cudnn-runtime-ubuntu22.04

To push the image:

	docker push quay.io/ucsc_cgl/kilosort4:12.4.1cudnn-runtime-ubuntu22.04
