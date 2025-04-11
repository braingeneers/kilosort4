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
        raise RuntimeError('Input filepath must start with s3://')

    torch.cuda.empty_cache()
    if mock:
        multisorting = mock_run_sorter(input_filepath)
    else:
        multisorting = ss.run_sorter(
            'kilosort4',
            read_maxwell(f'/tmp/{os.path.basename(input_filepath)}'),
            folder=output_dir,
            remove_existing_folder=True,
            verbose=True
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
