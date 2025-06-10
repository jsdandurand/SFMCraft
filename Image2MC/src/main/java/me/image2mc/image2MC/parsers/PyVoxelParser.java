package me.image2mc.image2MC.parsers;

import me.image2mc.image2MC.models.Pallette;
import me.image2mc.image2MC.models.Voxel;
import me.image2mc.image2MC.models.VoxelModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
public class PyVoxelParser implements VoxelParser {

  @Override
  public boolean supports(File file) {
    return file.getName().endsWith(".py");
  }

  // Doing this separately cause im lazy
  // TODO: Include this in the main parser regex
  private List<Pallette> parsePallete(File file) throws IOException {
    String content = Files.readString(file.toPath());

    int start = content.indexOf("pallette = [");
    int end = content.indexOf("]", start) + 2;

    String palleteString = content.substring(start + "pallette = ".length() + 1, end + 1);
    String[] palletteLines = palleteString.strip().split("\n");

    // Convert array to mutable list
    List<String> palletteList = new ArrayList<>(Arrays.asList(palletteLines));

    // Remove first and last lines
    if (!palletteList.isEmpty()) palletteList.removeFirst();
    if (!palletteList.isEmpty()) palletteList.removeLast();

    int parsed_lines = 0;
    List<Pallette> pallettes = new ArrayList<>();
    // We know there will be palletteList.size() lines.
    // Prefill the list with nulls so if a color fails to parse, it won't shift the indices
    for (int i = 0; i < palletteList.size(); i++) {
      pallettes.add(null);
    }

    for (int i = 0; i < palletteList.size(); i++) {
      String line = palletteList.get(i).strip();
      line = line.replace("{", "").trim();
      Pattern pattern = Pattern.compile("'red':\\s*(\\d+),\\s*'green':\\s*(\\d+),\\s*'blue':\\s*(\\d+)");
      Matcher matcher = pattern.matcher(line);

      if (matcher.find()) {
        int red = Integer.parseInt(matcher.group(1));
        int green = Integer.parseInt(matcher.group(2));
        int blue = Integer.parseInt(matcher.group(3));
        Pallette pallette = new Pallette(red, green, blue);
        pallettes.set(i, pallette);
        parsed_lines += 1;
      }
    }

    System.out.println("Parsed " + parsed_lines + " lines from the pallette.");
    return pallettes;
  }

  @Override
  public VoxelModel parse(File file) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(file));
    int width = 0, height = 0, depth = 0;
    List<Voxel> voxels = new ArrayList<>();
    List<Pallette> pallettes = this.parsePallete(file);
    Pattern dimPattern = Pattern.compile("(widthGrid|heightGrid|depthGrid)\\s*=\\s*(\\d+)");
    Pattern voxelPattern = Pattern.compile("\\{\\s*'x':\\s*(\\d+),\\s*'y':\\s*(\\d+),\\s*'z':\\s*(\\d+),\\s*'color':\\s*(\\d+)\\s*}");


    String line;
    int count = 0;
    while ((line = reader.readLine()) != null) {
      Matcher m = dimPattern.matcher(line);

      if (m.find()) {
        switch (m.group(1)) {
          case "widthGrid" -> width = Integer.parseInt(m.group(2));
          case "heightGrid" -> height = Integer.parseInt(m.group(2));
          case "depthGrid" -> depth = Integer.parseInt(m.group(2));
        }
      }

      m = voxelPattern.matcher(line);
      if (m.find()) {
        int x = Integer.parseInt(m.group(1));
        int y = Integer.parseInt(m.group(2));
        int z = Integer.parseInt(m.group(3));
        int color = Integer.parseInt(m.group(4));
        voxels.add(new Voxel(x, y, z, color));
      }
    }

    return new VoxelModel(width, height, depth, voxels, pallettes);
  }
}

