# SFMCraft: Real World to Minecraft Pipeline

Transform real-world photos and videos into stunning Minecraft builds using advanced 3D reconstruction techniques.

## Pipeline Demonstration

### Input → Structure from Motion → Voxel Grid → Minecraft Build

<table>
<tr>
<td align="center"><strong>Original Photos</strong></td>
<td align="center"><strong>Voxel representation</strong></td>
<td align="center"><strong>Minecraft Result</strong></td>
</tr>
<tr>
<td><img src="examples/fairlife/original.jpg" width="250" alt="Original photo"></td>
<td><img src="examples/fairlife/voxel.png" width="250" alt="Voxel representation"></td>
<td><img src="examples/minecraft_build.png" width="250" alt="Final Minecraft build"></td>
</tr>
</table>

*Example: Converting real-world architecture into a detailed Minecraft build*

### Step-by-Step Pipeline Visualization

<table>
<tr>
<td align="center"><strong>1. Video Input</strong></td>
<td align="center"><strong>2. Frame Sampling</strong></td>
<td align="center"><strong>3. Structure from Motion</strong></td>
</tr>
<tr>
<td><img src="examples/input_video.gif" width="200" alt="Input video walkthrough"></td>
<td><img src="examples/sampled_frames.jpg" width="200" alt="Extracted key frames"></td>
<td><img src="examples/sfm_reconstruction.png" width="200" alt="SfM camera poses and sparse points"></td>
</tr>
</table>

<table>
<tr>
<td align="center"><strong>4. Dense Point Cloud</strong></td>
<td align="center"><strong>5. Voxel Grid</strong></td>
<td align="center"><strong>6. Minecraft Build</strong></td>
</tr>
<tr>
<td><img src="examples/dense_pointcloud.png" width="200" alt="Dense 3D point cloud"></td>
<td><img src="examples/voxel_grid.png" width="200" alt="Voxelized representation"></td>
<td><img src="examples/final_build.png" width="200" alt="Complete Minecraft structure"></td>
</tr>
</table>

## How It Works

SFMCraft uses a sophisticated computer vision and 3D reconstruction pipeline to convert real-world scenes into Minecraft builds. The process involves several key stages:

### 1. Input Processing
The pipeline accepts either:
- **Multiple photographs** of a scene taken from different angles
- **Video footage** of a walkthrough or flythrough of the target area

For video input, the system intelligently samples key frames to ensure good coverage while avoiding redundant images.

### 2. Structure from Motion (SfM)
Using advanced photogrammetry techniques, the system:
- Detects and matches features across all input images
- Estimates camera positions and orientations for each photo
- Triangulates 3D points to create a sparse point cloud
- Reconstructs the 3D geometry of the photographed scene

This process essentially reverse-engineers the 3D structure that created the 2D photographs.

### 3. Dense Point Cloud Generation
The sparse SfM reconstruction is enhanced to create a dense, detailed point cloud:
- Multi-view stereo algorithms fill in gaps between sparse points
- Surface normals and colors are computed for each 3D point
- Noise filtering and outlier removal improve data quality

### 4. Voxelization
The continuous point cloud is converted into a discrete voxel grid:
- 3D space is divided into uniform cubic cells (voxels)
- Each voxel is classified as occupied or empty based on nearby points
- Voxel colors are determined by averaging nearby point cloud colors

### 5. Block Mapping and Optimization
The voxel grid is translated into Minecraft blocks:
- Each voxel's color is matched to the closest available Minecraft block
- A sophisticated color palette system ensures realistic material representation
- Block placement is optimized for structural integrity and visual appeal

### 6. Minecraft Integration
The final build is generated as:
- **Schematic files** for import into creative mode
- **Direct world generation** via the included Minecraft plugin
- **Command sequences** for automated building

## Technical Architecture

### Python Pipeline (`src/`)
- **`pipeline.py`**: Main orchestration and workflow management
- **`sfm.py`**: Structure from Motion implementation using COLMAP
- **`multi_scan_sfm.py`**: Multi-scan SfM for large scenes
- **`postprocess_pointcloud.py`**: Point cloud filtering and enhancement
- **`voxelize.py`**: Voxel grid generation from point clouds
- **`sample_video_frames.py`**: Intelligent frame extraction from videos
- **`view_model.py`**: 3D visualization and inspection tools

### Minecraft Plugin (`Image2MC/`)
- **Java-based Minecraft plugin** for Bukkit/Spigot servers
- **Block placement optimization** for large-scale builds
- **Real-time building** with progress tracking
- **Material palette management** for accurate color representation

## Key Features

- **High-Quality Reconstruction**: Advanced SfM algorithms ensure accurate 3D geometry
- **Intelligent Block Selection**: Sophisticated color matching for realistic material representation
- **Scalable Processing**: Handles everything from small objects to large architectural scenes
- **Multiple Input Formats**: Works with photos, videos, and existing point clouds
- **Minecraft Integration**: Seamless import into Minecraft worlds via plugin or schematics

## Applications

- **Architecture Visualization**: Convert real buildings into explorable Minecraft environments
- **Educational Projects**: Bring historical sites and landmarks into Minecraft classrooms
- **Creative Building**: Transform artistic inspiration from the real world into block form
- **Documentation**: Preserve important structures and spaces in an interactive digital format

---

*Transform your world, one block at a time.*

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
