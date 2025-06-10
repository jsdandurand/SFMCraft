package me.image2mc.image2MC.loaders;

import me.image2mc.image2MC.models.VoxelModel;
import me.image2mc.image2MC.parsers.VoxelParser;

import java.io.File;
import java.util.List;

public class VoxelLoader {
  private final List<VoxelParser> parsers;

  public VoxelLoader(List<VoxelParser> parsers) {
    this.parsers = parsers;
  }

  public VoxelModel load(File file) throws Exception {
    for (VoxelParser parser : parsers) {
      if (parser.supports(file)) {
        return parser.parse(file);
      }
    }
    throw new IllegalArgumentException("Unsupported voxel format: " + file.getName());
  }
}