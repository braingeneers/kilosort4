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


def zip_files(dir: str, zip_filename: str):
    with zipfile.ZipFile(zip_filename, 'w') as z:
        for root, dirs, files in os.walk(dir, topdown=False):
            for f in files:
                f = os.path.join(root, f)
                z.write(f, os.path.basename(f))


def mock_run_sorter(input_filepath: str):
    os.makedirs(output_dir, exist_ok=True)
    for file in ['a.npy', 'b.npy', 'c.npy']:
        with open(os.path.join(output_dir, file), 'w') as f:
            f.write(f'{file} has content: {file}')
    return input_filepath


def main(mock: bool = False):
    input_filepath: str = sys.argv[1]
    s3_client = boto3.client('s3', endpoint_url='https://s3-west.nrp-nautilus.io')

    if input_filepath.startswith('s3://'):
        print(f'Downloading {input_filepath} with boto3 version: {boto3.__version__}')
        bucket, key = input_filepath[len('s3://'):].split('/', maxsplit=1)
        s3_client.download_file(bucket, key, f'/tmp/{os.path.basename(input_filepath)}')
    else:
        raise RuntimeError(f'Input filepath must start with s3:// !  {input_filepath}')

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
    if mock:
        multisorting = mock_run_sorter(input_filepath)
    else:
        multisorting = ss.run_sorter(
            'kilosort4',
            read_maxwell(f'/tmp/{os.path.basename(input_filepath)}'),
            folder=output_dir,
            remove_existing_folder=True,
            verbose=True,
            # **params
        )
    print(multisorting)

    zip_filename = f'{os.path.basename(input_filepath)}.ks4.zip'
    print(f'Zipping files to: {zip_filename}')
    zip_files(dir=output_dir, zip_filename=zip_filename)

    if input_filepath.startswith('s3://braingeneers/ephys/'):
        uuid = input_filepath[len('s3://braingeneers/ephys/'):].split('/', maxsplit=1)[0]
        output_s3_key = f'ephys/{uuid}/derived/{zip_filename}'
        print(f'Uploading {zip_filename} to s3://braingeneers/{output_s3_key} with boto3 version: {boto3.__version__}')
        s3_client.upload_file(zip_filename, 'braingeneers', output_s3_key)
    else:
        print(f'Input filepath is unexpected and non-canonical ({input_filepath})!  Skipping upload to s3.', file=sys.stderr)


if __name__ == '__main__':
    main()
