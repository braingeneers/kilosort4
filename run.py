import torch
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.comparison as sc # Corrected import path
import spikeinterface.widgets as sw
import spikeinterface.preprocessing as sp
import spikeinterface.extractors as se
# Removed: import spikeinterface.postprocessing as spost
# Removed: import spikeinterface.qualitymetrics as sqm

# Import neo and SpikeData
import neo
import logging
import sys


import os
import zipfile
import boto3
import argparse
# Removed: import shutil

# Import plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import braingeneers.analysis.analysis as ba


LOG_FILE_NAME = "run_ks4.log"
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE_NAME, mode="a"),
                              stream_handler])


try:
    # Adjust this import path if necessary for your environment
  from spikedata.spikedata import SpikeData
  import spikedata.spikedata
  logging.info(f"SpikeData version loaded")

except ImportError:
    logging.exception("ERROR: Could not import SpikeData from braingeneers.data.datasets_electrophysiology.")
    logging.info("Please ensure the 'braingeneers.spikedata' package is installed and the path is correct.")
    sys.exit(1)



# --- Configuration ---
this_dir = os.path.dirname(__file__)
output_dir_ks4 = os.path.join(this_dir, 'output_ks4')
output_dir_ks2 = os.path.join(this_dir, 'output_ks2')
# Removed waveform directories
output_dir_comparison_plots = os.path.join(this_dir, 'comparison_plots')


# Ensure output directories exist
"""not sure if this is needed"""
os.makedirs(output_dir_ks4, exist_ok=True)
os.makedirs(output_dir_ks2, exist_ok=True)
os.makedirs(output_dir_comparison_plots, exist_ok=True) # Plot dir still needed


s3_client = boto3.client('s3', endpoint_url='https://s3-west.nrp-nautilus.io')
braingeneers_bucket = 'braingeneers'
kilosort4_version = 'kilosort4' # Use specific version name
kilosort2_version = 'kilosort2' # Use specific version name


# --- Helper Functions (Unchanged from previous version) ---

def zip_files(dir_to_zip: str, zip_filename: str) -> None:

    """Given a directory with multiple files, this will zip them into one file."""

    logging.info(f'Zipping files in {dir_to_zip} to: {zip_filename}')
    with zipfile.ZipFile(zip_filename, 'w') as z: 
        for root, _, files in os.walk(dir_to_zip, topdown=False):
            for f in files:
                file_path = os.path.join(root, f)
                z.write(file_path, os.path.basename(file_path))

    logging.info(f'Successfully created zip file: {zip_filename}')

def upload_local_to_s3(local_src: str, s3_dst_base: str, sorter_version: str) -> None:

    """
    Given a local filepath and a base S3 destination URI (up to the UUID),
    this will upload it to the correct derived data path.
    """
    if not s3_dst_base.startswith(f's3://{braingeneers_bucket}/ephys/'):
        logging.exception(f'Base S3 destination URI is unexpected ({s3_dst_base})! Skipping upload.')
        return
        
    try:
        prefix = f's3://{braingeneers_bucket}/ephys/'
        uuid_and_rest = s3_dst_base[len(prefix):]
        uuid = uuid_and_rest.split('/', maxsplit=1)[0]
        s3_key = f'ephys/{uuid}/derived/{sorter_version}/{local_src}'
        full_s3_uri = f's3://{braingeneers_bucket}/{s3_key}'
    except Exception as e:
        logging.exception(f"Error constructing S3 key from base destination {s3_dst_base}: {e}")
        return

    logging.info(f'Uploading {local_src} to {full_s3_uri} using boto3 version: {boto3.__version__}')
    try:
        s3_client.upload_file(local_src, braingeneers_bucket, s3_key)
        logging.info(f'Successfully uploaded to {full_s3_uri}')
    except Exception as e:
        logging.exception(f'Error uploading {local_src} to {full_s3_uri}: {e}')

def download_s3_to_local(s3_src: str, local_dst: str) -> None:

    """Given an s3 URI, this will download it locally."""
    
    if not s3_src.startswith('s3://'):
        raise RuntimeError(f'Input filepath must start with s3:// ! Got: {s3_src}')
    logging.info(f'Downloading {s3_src} to {local_dst} using boto3 version: {boto3.__version__}')
    try:
        bucket, key = s3_src[len('s3://'):].split('/', maxsplit=1)
        s3_client.download_file(bucket, key, local_dst)
        logging.info(f'Successfully downloaded {s3_src} to {local_dst}')
    except Exception as e:
        logging.exception(f'Error downloading {s3_src}: {e}')
        raise

def s3_destination(s3_src_uri: str, filename: str, kilosort_ver: str):

    """From an input s3 URI, determine the s3 destination URI where kilosort outputs should go."""
    if not s3_src_uri.startswith(f's3://{braingeneers_bucket}/ephys/'):
        logging.exception(f'URI is unexpected and non-canonical ({s3_src_uri})!  Skipping upload to s3.')

    uuid = s3_src_uri[len(f's3://{braingeneers_bucket}/ephys/'):].split('/', maxsplit=1)[0]
    return f's3://{braingeneers_bucket}/ephys/{uuid}/derived/{kilosort_ver}/{filename}'


def unzip_file(zip_filepath: str, dest_dir: str) -> str:
    """Given a zip file, this will unzip it into a destination directory."""
    logging.info(f'Unzipping {zip_filepath} to: {dest_dir}')
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_filepath, 'r') as z:
        z.extractall(dest_dir)
    logging.info(f'Successfully unzipped to: {dest_dir}')
    return dest_dir # Return the path to the unzipped directory

def mock_run_sorter(input_filepath: str, output_dir: str) -> si.BaseSorting:
    """Creates mock output files and returns a dummy sorting object."""
    logging.info(f"--- MOCK RUNNING SORTER on {input_filepath} -> {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)
    mock_files = {
        'spike_times.npy': np.array([100, 250, 500, 150, 300, 600], dtype=np.int64),
        'spike_clusters.npy': np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        'amplitudes.npy': np.random.rand(6),
        'pc_features.npy': np.random.rand(6, 3, 2),
        'template_features.npy': np.random.rand(6, 2),
        'channel_map.npy': np.arange(10),
        'channel_positions.npy': np.random.rand(10, 2) * 100,
        'cluster_info.tsv': 'cluster_id\tgroup\tch\tfr\tx\ty\n0\tgood\t5\t10.5\t20\t30\n1\tmua\t8\t5.2\t50\t60\n', # Added pos
        'params.py': 'dat_path = "mock"\nsample_rate = 30000\n'
    }
    for filename, content in mock_files.items():
        filepath = os.path.join(output_dir, filename)
        if isinstance(content, str):
            with open(filepath, 'w') as f: f.write(content)
        elif isinstance(content, np.ndarray): np.save(filepath, content)
    try:
        # Kilosort reader should pick up cluster_info.tsv potentially
        mock_sorting = si.read_kilosort(output_dir, keep_good_only=False)
        logging.info(f"Loaded mock Kilosort sorting: {mock_sorting.get_unit_ids()}")
        # Attempt to add position property if cluster_info was read correctly
        if 'x' in mock_sorting.get_property_keys() and 'y' in mock_sorting.get_property_keys():
             x_coords = mock_sorting.get_property('x')
             y_coords = mock_sorting.get_property('y')
             locations = np.vstack((x_coords, y_coords)).T
             mock_sorting.set_property('location', locations)
             logging.info("Added 'location' property to mock sorting from cluster_info.")

    except Exception as e:
        logging.info(f"Could not load mock Kilosort data properly, creating basic NumpySorting: {e}")
        sampling_frequency = 30000
        times = mock_files['spike_times.npy']
        labels = mock_files['spike_clusters.npy']
        units_dict = {unit_id: times[labels == unit_id] for unit_id in np.unique(labels)}
        mock_sorting = si.NumpySorting.from_dict([units_dict], sampling_frequency=sampling_frequency)

    logging.info(f"--- MOCK SORTER RUN COMPLETE for {input_filepath} ---")
    return mock_sorting


# --- Kilosort Processing Functions (Unchanged) ---

def process_with_kilosort(input_filepath: str, output_dir: str, mock: bool = False) -> si.BaseSorting:
    
    """Accepts a local Maxwell h5 file as an input and produces a few dozen (mostly numpy array) files in an output dir"""

    if mock: return mock_run_sorter(input_filepath, output_dir)

    logging.info(f'Running {kilosort4_version} -> output dir: {output_dir}')

    params_ks4 = {"acg_threshold": 0.1,
              # algorithm lloyd
              "artifact_threshold": 1949.48,
              "batch_size": 6000,
              "binning_depth": 5,
              "ccg_threshold": 0.1,
              "cluster_downsampling": 20,
            #   "cluster_pcs": 64, # Temporarily removed Kilosort 4 parameter
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
              "nblocks": 0,
              "nearest_chans": 20,
              "nearest_templates": 69,
              "nskip": 25,
              "nt": 61,
              "nt0min": 6,
              # power_iteration_normalizer auto
              # random_state null
              # run_arg_0
              # run_arg_1
            #   "scaleproc": 200, # Temporarily removed Kilosort 4 parameter
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
              # Add other valid parameters from the traceback if needed and known
              # Example: 'fs', 'shift', 'scale', 'highpass_cutoff', 'drift_smoothing',
              # 'max_channel_distance', 'max_peels', 'x_centers', 'duplicate_spike_ms',
              # 'position_limit', 'save_extra_vars', 'save_preprocessed_copy',
              # 'torch_device', 'bad_channels', 'clear_cache', 'use_binary_file',
              # 'delete_recording_dat', 'pool_engine', 'n_jobs', 'chunk_duration',
              # 'progress_bar', 'mp_context', 'max_threads_per_worker'
              }
    logging.info(f"Using Kilosort 4 parameters: {params_ks4}")
    torch.cuda.empty_cache()
    try:
        # Read the Maxwell file
        recording = se.read_maxwell(input_filepath)

        # --- Add logging for recording properties ---
        logging.info(f"Recording loaded: {recording}")
        logging.info(f"Recording sampling frequency: {recording.get_sampling_frequency()} Hz")
        logging.info(f"Recording number of channels: {recording.get_num_channels()}")
        logging.info(f"Recording channel IDs: {recording.get_channel_ids()}")
        # --- End logging for recording properties ---
 # --- Explicitly select data channels ---
        # Assuming channels '0' through '1019' are data channels (1020 total)
        # and channels '1020' through '1023' are non-data channels.
        all_channel_ids = recording.get_channel_ids()
        data_channel_ids = [cid for cid in all_channel_ids if int(cid) < 1020]
        recording_filtered = recording.select_channels(channel_ids=data_channel_ids)

        logging.info(f"Filtered recording has {recording_filtered.get_num_channels()} data channels.")
        logging.info(f"Filtered recording channel IDs: {recording_filtered.get_channel_ids()[:5]} ... {recording_filtered.get_channel_ids()[-5:]}") # Log first/last few


        sorting = ss.run_sorter(kilosort4_version, 
                                recording_filtered,
                                folder=output_dir,
                                remove_existing_folder=True, 
                                verbose=True, 
                                **params_ks4)
        logging.info(f"{kilosort4_version} sorting complete. Output units: {sorting.get_unit_ids()}")
        return sorting
    except Exception as e:
        logging.exception(f"Error running {kilosort4_version}: {e}"); raise


# --- Position Retrieval Function ---


def get_neuron_positions(sd, neuron_index=None):
    """
    Extract neuron positions from a SpikeData object.

    This function checks if the SpikeData object's neuron_data (or neuron_attributes)
    is a dictionary in the dictionary‐of‐lists format (i.e. with keys such as "position"
    mapping to lists of values). If so, it returns the positions as a NumPy array.
    Otherwise, it falls back to iterating over the items.

    Parameters:
        sd : SpikeData
            A SpikeData object that should have either 'neuron_data' or 'neuron_attributes'.
        neuron_index : int, optional
            If provided, returns only the position for that neuron.

    Returns:
        np.ndarray or tuple:
            If neuron_index is None, returns an array of shape (N, 2) where each row is [x, y];
            otherwise, returns the (x, y) tuple for the specified neuron.
    """
    positions = None

    # First, check neuron_data
    if hasattr(sd, 'neuron_data') and sd.neuron_data:
        # If neuron_data is a dict
        if isinstance(sd.neuron_data, dict):
            # Check if it's in dictionary-of-lists format (e.g., {"position": [...], ...})
            if "position" in sd.neuron_data:
                positions = np.array(sd.neuron_data["position"])
            else:
                # Otherwise assume it's a dict of per-neuron metadata dictionaries.
                positions = []
                for key, meta in sd.neuron_data.items():
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
                positions = np.array(positions)
        # If neuron_data is a list, assume each element is a metadata dictionary.
        elif isinstance(sd.neuron_data, list):
            positions = []
            for meta in sd.neuron_data:
                if isinstance(meta, dict):
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
            positions = np.array(positions)
    # Fallback: check neuron_attributes if neuron_data is not available.
    elif hasattr(sd, 'neuron_attributes') and sd.neuron_attributes:
        if isinstance(sd.neuron_attributes, dict):
            if "position" in sd.neuron_attributes:
                positions = np.array(sd.neuron_attributes["position"])
            else:
                positions = []
                for key, meta in sd.neuron_attributes.items():
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
                positions = np.array(positions)
        elif isinstance(sd.neuron_attributes, list):
            positions = []
            for meta in sd.neuron_attributes:
                if isinstance(meta, dict):
                    pos = meta.get("position", None)
                    if pos is not None:
                        positions.append(pos)
            positions = np.array(positions)

    if positions is None or positions.size == 0:
        raise ValueError("No valid neuron positions found in SpikeData object.")
    
    if neuron_index is not None:
        if neuron_index < 0 or neuron_index >= positions.shape[0]:
            raise IndexError("Neuron index out of bounds.")
        return positions[neuron_index]
    
    return positions


# --- Plotting Function ---
def plot_matched_neuron_pair(ks2_unit_id, ks4_unit_id,
                             sd_ks2: SpikeData, sd_ks4: SpikeData,
                             plot_output_dir):
    """
    Plots the positions of a matched pair of KS2 and KS4 units,
    retrieving positions from SpikeData objects using unit IDs.

    Parameters:
        ks2_unit_id : int/str
            Unit ID from Kilosort 2.
        ks4_unit_id : int/str
            Unit ID from Kilosort 4 matched to ks2_unit_id.
        sd_ks2 : SpikeData
            SpikeData object for Kilosort 2 results.
        sd_ks4 : SpikeData
            SpikeData object for Kilosort 4 results.
        plot_output_dir : str
            Directory to save the plot.
    """
    # Retrieve positions using the dedicated function
    pos_ks2 = get_neuron_positions(sd_ks2, ks2_unit_id)
    pos_ks4 = get_neuron_positions(sd_ks4, ks4_unit_id)

    if pos_ks2 is None:
        logging.info(f"Warning: Could not find position for KS2 unit {ks2_unit_id}. Skipping plot.")
        return
    if pos_ks4 is None:
        logging.info(f"Warning: Could not find position for KS4 unit {ks4_unit_id}. Skipping plot.")
        return

    try:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot KS2 position
        ax.scatter(pos_ks2[0], pos_ks2[1], c='blue', marker='o', s=100, label=f'KS2 Unit {ks2_unit_id}', alpha=0.8, edgecolors='k')
        # Plot KS4 position
        ax.scatter(pos_ks4[0], pos_ks4[1], c='red', marker='s', s=100, label=f'KS4 Unit {ks4_unit_id}', alpha=0.8, edgecolors='k')

        # Draw a line connecting them
        ax.plot([pos_ks2[0], pos_ks4[0]], [pos_ks2[1], pos_ks4[1]], 'k--', alpha=0.5)

        # Calculate distance
        distance = np.linalg.norm(pos_ks2 - pos_ks4)

        # Find overall min/max for padding
        all_x = [pos_ks2[0], pos_ks4[0]]
        all_y = [pos_ks2[1], pos_ks4[1]]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = max(x_max - x_min, 10) # Ensure minimum range
        y_range = max(y_max - y_min, 10)
        padding_x = x_range * 0.3 # Increased padding slightly
        padding_y = y_range * 0.3

        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

        ax.set_xlabel("X Position (µm)")
        ax.set_ylabel("Y Position (µm)")
        ax.set_title(f"Matched Unit Locations: KS2 ({ks2_unit_id}) vs KS4 ({ks4_unit_id})\nDistance: {distance:.2f} µm")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_aspect('equal', adjustable='box') # Ensure equal scaling

        plot_filename = os.path.join(plot_output_dir, f'match_ks2_{ks2_unit_id}_ks4_{ks4_unit_id}.png')
        plt.savefig(plot_filename)
        plt.close(fig)
        # print(f"Saved comparison plot: {plot_filename}")

    except Exception as e:
        logging.exception(f"Error plotting pair KS2({ks2_unit_id})-KS4({ks4_unit_id}): {e}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run Kilosort, optionally compare KS2 and KS4, read output as SpikeData, and plot matched unit locations.")
    parser.add_argument("input_uri", help="S3 URI of the input Maxwell H5 file (e.g., s3://braingeneers/ephys/...)")
    parser.add_argument("-c", "--compare", action="store_true", help="Run both KS2 and KS4 and compare results, including plotting matched unit locations.")
    parser.add_argument("-m", "--mock", action="store_true", help="Use mock sorter runs instead of actual Kilosort (for testing).")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading results to S3.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating comparison plots (if --compare is active).")
    parser.add_argument("--ks2-zip-uri", help="S3 URI of the pre-sorted Kilosort 2 output zip file (required if --compare is used).",
                        default="s3://braingeneers/ephys/2024-01-05-e-uploader-test/derived/kilosort2/7month_2953_ks2_phy.zip") # Added default KS2 zip URI
    parser.add_argument("--ks4-zip-uri", help="S3 URI of the pre-sorted Kilosort 4 output zip file (if provided, a new KS4 sorting will not be run).")


    # Removed --keep-waveforms

    args = parser.parse_args()

    input_uri: str = args.input_uri
    uuid = input_uri.split("/")[4]
    base_filename = os.path.basename(input_uri).split('.')[0]
    local_filepath: str = f'/tmp/{os.path.basename(input_uri)}'

    # --- Download Data ---
    try:
        download_s3_to_local(s3_src=input_uri, local_dst=local_filepath)
    except Exception:
        sys.exit(1)


    # --- Process Kilosort 4 (Run new or Load from Zip) ---
    ks4_sorting: si.BaseSorting = None
    ks4_spikedata: SpikeData = None
    ks4_output_dir: str = None # Variable to store the path to the KS4 output directory

        # Run a new Kilosort 4 sorting
    logging.info("\n--- Running Kilosort 4 ---")
    try:
        # Use the raw recording for KS4
        ks4_sorting = process_with_kilosort(local_filepath, output_dir_ks4, mock=args.mock)
        zip_files(dir_to_zip=output_dir_ks4, zip_filename=os.path.join(this_dir, f'{base_filename}_ks4_output.zip'))
        ks4_spikedata = ba.load_spike_data(uuid=uuid, full_path=os.path.join(this_dir, f'{base_filename}_ks4_output.zip'))


        logging.info(f"KS4 SpikeData object created.")
            # print(f"KS4 SpikeData neuron_data keys: {getattr(ks4_spikedata, 'neuron_data', {}).keys()}")

    except Exception as e:
        logging.exception(f"Failed to run or process Kilosort 4: {e}")
        ks4_sorting = None # Ensure it's None on failure


    # --- Process Kilosort 2 (Download and Unzip Pre-sorted Output) ---
    ks2_sorting: si.BaseSorting = None
    ks2_spikedata: SpikeData = None
    ks2_output_dir: str = None # Variable to store the path to the unzipped KS2 output

    if args.compare:
        ks2_zip_uri = args.ks2_zip_uri
        if not ks2_zip_uri:
             logging.error("Comparison requested but --ks2-zip-uri was not provided.")
        else:
            logging.info(f"\n--- Downloading and Unzipping Pre-sorted Kilosort 2 Output from {ks2_zip_uri} ---")
            local_ks2_zip_filepath = f'/tmp/{os.path.basename(ks2_zip_uri)}'
            # Download the KS2 zip file
            download_s3_to_local(s3_src=ks2_zip_uri, local_dst=local_ks2_zip_filepath)

            # Read the unzipped Kilosort2 output using spikeinterface
            # Assuming the unzipped structure is compatible with si.read_kilosort

            ks2_spikedata= ba.load_spike_data(uuid=uuid, full_path=local_ks2_zip_filepath)
            logging.info(f"KS2 SpikeData object created.")


    # --- Compare Results & Plot Locations ---
    comparison: sc.BaseComparison = None # Store comparison object
    # Check if both sorters have valid sorting objects and SpikeData objects for comparison
    if ks4_sorting and ks2_sorting and ks4_spikedata and ks2_spikedata:
        logging.info("\n--- Comparing KS2 and KS4 Results ---")
        try:
            comparison = sc.compare_two_sorters(
                sorting1=ks2_sorting, sorting2=ks4_sorting,
                sorting1_name='kilosort2', sorting2_name='kilosort4',
                verbose=True
            )

            logging.info("Plotting agreement matrix...")
            try:
                fig, ax = plt.subplots()
                sw.plot_agreement_matrix(comparison=comparison, axes=ax)
                agreement_fig_path = os.path.join(this_dir, f'{base_filename}_ks2_ks4_agreement.png')
                plt.savefig(agreement_fig_path)
                logging.info(f"Saved agreement matrix to {agreement_fig_path}")
                plt.close(fig)
            except Exception as plot_err:
                    logging.info(f"Error plotting agreement matrix: {plot_err}")

            logging.info("\nMatch Counts (KS2 vs KS4):"); logging.info(comparison.match_event_count)
            logging.info("\nAgreement Scores (KS2 vs KS4):"); logging.info(comparison.agreement_scores)

            # --- Plot Matched Pairs ---
            if not args.skip_plots:
                logging.info("\n--- Plotting Matched Unit Locations (using SpikeData) ---")
                map_1_to_2 = dict(comparison.best_match_12) # KS2 -> KS4 map
                y = []
                for x, z in map_1_to_2.items():
                    y.append([int(x), int(z)])
                map_1_to_2 = dict(y)
                
                plot_count = 0
                not_found_count = 0
                for ks2_id, ks4_id in map_1_to_2.items():
                    if ks4_id != -1: # Check if KS2 unit has a match in KS4
                        # Plot using the SpikeData objects and original IDs
                        plot_matched_neuron_pair(ks2_id, ks4_id,
                                                    ks2_spikedata, ks4_spikedata,
                                                    output_dir_comparison_plots)
                        # Simple check if files were created (plot_matched_neuron_pair handles internal errors)
                        plot_filename = os.path.join(output_dir_comparison_plots, f'match_ks2_{ks2_id}_ks4_{ks4_id}.png')
                        if os.path.exists(plot_filename):
                            plot_count += 1
                        else:
                            # This implies position wasn't found in get_neuron_position_by_id
                            not_found_count +=1

                logging.info(f"Attempted to generate {plot_count + not_found_count} matched unit location plots.")
                logging.info(f"Successfully generated {plot_count} plots in '{output_dir_comparison_plots}'.")
                if not_found_count > 0:
                        logging.info(f"Warning: Could not find positions for {not_found_count} matched pairs within SpikeData objects.")
            else:
                logging.info("\n--- Skipping Matched Unit Location Plotting as requested ---")

        except Exception as comp_err:
            logging.exception(f"Error during comparison or plotting setup: {comp_err}")

    else:
         # Print more specific reasons for skipping comparison
            reasons = []
    if not args.compare: reasons.append("--compare flag not used")
    if args.compare and not ks4_sorting: reasons.append("KS4 sorter failed or found no units")
    if args.compare and not ks2_sorting: reasons.append("KS2 sorting could not be loaded from zip")
    if args.compare and ks4_sorting and not ks4_spikedata: reasons.append("KS4 SpikeData object creation failed")
    if args.compare and ks2_sorting and not ks2_spikedata: reasons.append("KS2 SpikeData object creation failed")

    logging.info(f"\n--- Skipping Comparison and Plotting: {' '.join(reasons)} ---")



    # --- Zip and Upload Results ---
    if not args.skip_upload:
        logging.info("\n--- Zipping and Uploading Results ---")
        s3_base_uri = s3_destination(input_uri, base_filename, kilosort4_version)

        
        # Zip/Upload comparison plots (if compare and not skipped)
        if args.compare and not args.skip_plots and os.path.exists(output_dir_comparison_plots) and any(os.scandir(output_dir_comparison_plots)):
            zip_filename_plots = f'{base_filename}.ks2_ks4_comparison_plots.zip'
            local_zip_path_plots = os.path.join(this_dir, zip_filename_plots)
            try:
                zip_files(dir_to_zip=output_dir_comparison_plots, zip_filename=local_zip_path_plots)
                upload_local_to_s3(local_src=local_zip_path_plots, s3_dst_base=s3_base_uri, sorter_version=f"{kilosort2_version}_{kilosort4_version}_comparison")
            except Exception as e: logging.exception(f"Error zipping/uploading comparison plots: {e}")
        elif args.compare and not args.skip_plots: logging.info("Comparison plots directory empty/missing. Skipping zip/upload.")

    else:
        logging.info("\n--- Skipping S3 Upload as requested ---")

    # --- Cleanup ---
    logging.info("\n--- Cleaning up temporary local file ---")
    try:
        if os.path.exists(local_filepath):
            os.remove(local_filepath)
            logging.info(f"Removed temporary file: {local_filepath}")
    except OSError as e:
        logging.info(f"Error removing temporary raw data file {local_filepath}: {e}")

    # Clean up the temporary unzipped KS2 directory if it exists
    if ks2_output_dir and os.path.exists(ks2_output_dir):
        try:
            shutil.rmtree(ks2_output_dir)
            logging.info(f"Removed temporary unzipped KS2 output directory: {ks2_output_dir}")
        except OSError as e:
            logging.info(f"Error removing temporary unzipped KS2 output directory {ks2_output_dir}: {e}")

# Clean up the temporary unzipped KS4 directory if it exists (only if loaded from zip)
    if args.ks4_zip_uri and ks4_output_dir and os.path.exists(ks4_output_dir):
        try:
            shutil.rmtree(ks4_output_dir)
            logging.info(f"Removed temporary unzipped KS4 output directory: {ks4_output_dir}")
        except OSError as e:
            logging.info(f"Error removing temporary unzipped KS4 output directory {ks4_output_dir}: {e}")


    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    # Dependency check
    try:
        import neo
        # Test import again
        from spikedata.spikedata import SpikeData
        import matplotlib
        import numpy
        import shutil
        

    except ImportError as e:
        logging.exception(f"Missing required library: {e.name}")
        logging.exception("Please install necessary libraries:")
        logging.exception("pip install spikeinterface[full] neo braingeneers.spikedata matplotlib numpy")
        sys.exit(1)

    main()