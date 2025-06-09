# Image2MC

Convert real-world objects to Minecraft builds using Structure from Motion and voxelization.

## Setup

1. Install COLMAP:
```bash
# For Ubuntu/Debian
sudo apt-get install colmap

# For other systems, see: https://colmap.github.io/install.html
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
- `src/`: Source code
  - `sfm.py`: Structure from Motion processing
  - `voxelize.py`: Point cloud to voxel conversion
  - `minecraft.py`: Voxel to Minecraft block conversion
- `data/`: Directory for input images and intermediate results
- `output/`: Final Minecraft build output

## Usage
1. Place your input images in `data/images/`
2. Run the SfM pipeline:
```bash
python src/sfm.py --input data/images --output data/sfm_output
```
