package me.image2mc.image2MC.parsers;
import me.image2mc.image2MC.models.VoxelModel;

import java.io.File;

public interface VoxelParser {
  boolean supports(File file);
  VoxelModel parse(File file) throws Exception;
}