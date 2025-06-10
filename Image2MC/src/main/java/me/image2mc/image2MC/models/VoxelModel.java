package me.image2mc.image2MC.models;
import java.util.List;

public record VoxelModel(int width, int height, int depth, List<Voxel> voxels, List<Pallette> palletes) {}