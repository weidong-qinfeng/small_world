# HMI analysis

Analysis code for generating the HMI zebrafish EM connectivity figures and graph summary statistics.

## Repository structure

```text
HMI_analysis/
├── data/
├── results/
├── scripts/
├── src/
├── environment.yml
├── pyproject.toml
└── README.md
```

## Setup

Create the conda environment from the project root:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate hmi_analysis
```

The environment file installs the local `zfish` package in editable mode. If needed, this can also be run manually from the project root:

```bash
pip install -e .
```

Test that the package can be imported:

```bash
python - <<'PY'
from zfish.labels import class_label, neurotransmitter_color, parse_network_levels

print(class_label("1"))
print(neurotransmitter_color("VGluT2"))
print(parse_network_levels("['output_one', 'network_seed']"))
PY
```

Expected output:

```text
I
#00DF00
['output_one', 'network_seed']
```

## Data files

The main dataframe is expected at:

```text
data/em_zfish1_dataframe.xlsx
```

The PyVis layout file is expected at:

```text
data/pyvis_node_positions.json
```

The voxel script uses cached CAVE soma data by default:

```text
data/cave_somas_in_big_box.csv
```

If this cache file is present, the voxel script can run without querying CAVE.

## CAVE authentication

CAVE authentication is only required if the soma cache needs to be refreshed.

To create and save a CAVE token, start Python:

```bash
python
```

Then run:

```python
import caveclient

client = caveclient.CAVEclient(
    server_address="https://global.brain-wire-test.org"
)

client.auth.get_new_token(open=True)
```

Copy the token from the browser page, then save it:

```python
client.auth.save_token(token="PASTE_TOKEN_HERE", overwrite=True)
```

Exit Python:

```python
exit()
```

Do not store CAVE tokens in this repository.

## Run individual scripts

Generate the connectivity matrix:

```bash
python scripts/connectivity_matrix.py
```

Generate the interactive PyVis graph:

```bash
python scripts/pyvis_graph.py
```

Generate voxelized z-slice figures using the cached CAVE soma file:

```bash
python scripts/voxel.py
```

Refresh the CAVE soma cache and then generate the voxelized z-slice figures:

```bash
python scripts/voxel.py --refresh-cache
```

Generate graph summary statistics:

```bash
python scripts/graph_summary.py
```

## Run all scripts

From the project root:

```bash
python scripts/connectivity_matrix.py
python scripts/pyvis_graph.py
python scripts/voxel.py
python scripts/graph_summary.py
```

Or as a loop:

```bash
for script in scripts/connectivity_matrix.py scripts/pyvis_graph.py scripts/voxel.py scripts/graph_summary.py; do
    python "$script"
done
```

## Expected outputs

```text
results/connectivity_matrix/connectivity_matrix.svg
results/pyvis/pyvis.html
results/voxel/z_slice_01.svg
results/voxel/z_slice_02.svg
results/voxel/z_slice_03.svg
results/voxel/z_slice_04.svg
results/voxel/z_slice_05.svg
results/voxel/z_slice_06.svg
results/voxel/z_slice_07.svg
results/voxel/z_slice_08.svg
results/voxel/z_slice_09.svg
results/voxel/z_slice_10.svg
results/voxel/z_slice_11.svg
results/voxel/collapsed_xy_slices_01_to_06.svg
```

The graph summary script prints summary statistics to the terminal.

## Recreate the environment

If the environment needs to be recreated from scratch:

```bash
conda deactivate
conda env remove -n hmi_analysis
conda env create -f environment.yml
conda activate hmi_analysis
```