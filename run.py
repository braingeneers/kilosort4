import torch

import spikeinterface as si
import spikeinterface.sorters as ss

import os
import sys
import zipfile
import boto3

from kilosort import io, run_kilosort
from spikeinterface.extractors import read_nwb_recording, read_maxwell


this_dir = os.path.dirname(__file__)
output_dir = os.path.join(this_dir, 'output')

s3_client = boto3.client('s3', endpoint_url='https://s3-west.nrp-nautilus.io')
braingeneers_bucket = 'braingeneers'
kilosort_version = 'kilosort4'


def zip_files(dir: str, zip_filename: str) -> None:
    """Given a directory with multiple files, this will zip them into one file."""
    print(f'Zipping files in {dir} to: {zip_filename}')
    with zipfile.ZipFile(zip_filename, 'w') as z:
        for root, dirs, files in os.walk(dir, topdown=False):
            for f in files:
                f = os.path.join(root, f)
                z.write(f, os.path.basename(f))


def upload_local_to_s3(src: str, dst: str) -> None:
    """Given a local filepath, this will upload it to an s3 URI."""
    if not dst.startswith(f's3://{braingeneers_bucket}/ephys/'):
        print(f'URI is unexpected and non-canonical ({dst})!  Skipping upload to s3.', file=sys.stderr)

    uuid = dst[len(f's3://{braingeneers_bucket}/ephys/'):].split('/', maxsplit=1)[0]
    dst_s3_key = f'ephys/{uuid}/derived/{kilosort_version}/{src}'
    print(f'Uploading {src} to {dst} with boto3 version: {boto3.__version__}')
    s3_client.upload_file(src, braingeneers_bucket, dst[len(f's3://{braingeneers_bucket}/'):])


def download_s3_to_local(src: str, dst: str) -> None:
    """Given an s3 URI, this will download it locally."""
    if not src.startswith('s3://'):
        raise RuntimeError(f'Input filepath must start with s3:// !  {src}')

    print(f'Downloading {src} to {dst} with boto3 version: {boto3.__version__}')
    bucket, key = src[len('s3://'):].split('/', maxsplit=1)
    s3_client.download_file(bucket, key, dst)


def s3_destination(src: str, filename: str):
    """From an input s3 URI, determine the s3 destination URI where kilosort outputs should go."""
    if not src.startswith(f's3://{braingeneers_bucket}/ephys/'):
        print(f'URI is unexpected and non-canonical ({src})!  Skipping upload to s3.', file=sys.stderr)

    uuid = src[len(f's3://{braingeneers_bucket}/ephys/'):].split('/', maxsplit=1)[0]
    return f's3://{braingeneers_bucket}/ephys/{uuid}/derived/{kilosort_version}/{filename}'


def mock_run_sorter(input_filepath: str) -> str:
    """Writes three text files with some content into an output dir."""
    os.makedirs(output_dir, exist_ok=True)
    for file in ['a.npy', 'b.npy', 'c.npy']:
        with open(os.path.join(output_dir, file), 'w') as f:
            f.write(f'{file} has content: {file}')
    return input_filepath


def process_with_kilosort(input_filepath: str, mock: bool = False) -> str:
    """
    Accepts a local Maxwell h5 file as an input and produces a few dozen (mostly numpy array) files
    in an output dir.
    """
    if mock:
        return mock_run_sorter(input_filepath)

    params = {"acg_threshold": 0.1,
              # algorithm lloyd
              "artifact_threshold": 1949.48,
              "batch_size": 6000,
              "binning_depth": 5,
              "ccg_threshold": 0.1,
              "cluster_downsampling": 20,
              "cluster_pcs": 64,
              # copy_x True
              "dmin": 66.002,
              "dminx": 17.5,
              "do_CAR": True,
              "do_correction": False,
              # init k-means++
              "invert_sign": False,
              "keep_good_only": False,
              # max_channel_distance 36.835
              # max_iter 300
              "min_template_size": 18.9638,
              # n_clusters 7
              # n_components 3
              # n_init 10
              # n_iter 5
              # n_oversamples 10
              "n_pcs": 3,
              "n_templates": 7,
              "nblocks": 1,
              "nearest_chans": 20,
              "nearest_templates": 69,
              "nskip": 25,
              "nt": 61,
              "nt0min": 6,
              # power_iteration_normalizer auto
              # random_state null
              # run_arg_0
              # run_arg_1
              "scaleproc": 200,
              "sig_interp": 20,
              "skip_kilosort_preprocessing": False,
              "template_sizes": 8,
              "templates_from_data": True,
              "Th_learned": 8,
              "Th_single_ch": 6,
              "Th_universal": 9,
              # tol 0.0001
              # verbose 0
              "whitening_range": 32,
              }

    torch.cuda.empty_cache()
    return ss.run_sorter(
        kilosort_version,
        read_maxwell(input_filepath),
        folder=output_dir,
        remove_existing_folder=True,
        verbose=True,
        # **params
    )


def main(mock: bool = False):
    input_uri: str = sys.argv[1]
    zip_filename: str = f'{os.path.basename(input_uri)}.ks4.zip'
    local_filepath: str = f'/tmp/{os.path.basename(input_uri)}'

    download_s3_to_local(src=input_uri, dst=local_filepath)
    process_with_kilosort(local_filepath, mock=mock)
    zip_files(dir=output_dir, zip_filename=zip_filename)
    upload_local_to_s3(src=zip_filename, dst=s3_destination(input_uri, zip_filename))


if __name__ == '__main__':
    main()
