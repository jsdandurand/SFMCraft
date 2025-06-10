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
    System.out.println(palletteLines[palletteLines.length - 1]);
    System.out.println(palletteLines[palletteLines.length - 2]);
    System.out.println(palletteLines[0]);
    System.out.println("Size of thingy " + palletteLines.length);

    for (String line: palletteLines) {
      line = line.replace("{", "")
          .trim();
    }
    return null;
  }

  @Override
  public VoxelModel parse(File file) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(file));
    int width = 0, height = 0, depth = 0;
    List<Voxel> voxels = new ArrayList<>();
    this.parsePallete(file);
    Pattern dimPattern = Pattern.compile("(widthGrid|heightGrid|depthGrid)\\s*=\\s*(\\d+)");
    Pattern voxelPattern = Pattern.compile("\\{\\s*'x':\\s*(\\d+),\\s*'y':\\s*(\\d+),\\s*'z':\\s*(\\d+),\\s*'color':\\s*(\\d+)\\s*}");
    Pattern palletteStart = Pattern.compile("^\\s*pallette\\s*=.*$");
    Pattern rgbLine = Pattern.compile("^\\s*\\{\\s*'red'\\s*:\\s*\\d+\\s*,\\s*'green'\\s*:\\s*\\d+\\s*,\\s*'blue'\\s*:\\s*\\d+\\s*\\},\\s*#\\s*\\d+\\s*$\n");

    boolean isInPallette = false;
    String line;
    int count = 0;
    while ((line = reader.readLine()) != null) {
      Matcher m = dimPattern.matcher(line);

      // Assume pallette is at the end
      // Just use up all the remaining lines
      Matcher rgb = rgbLine.matcher(line);
      if (rgb.find()){
        count += 1;
        System.out.println("THE COUNT IS THE FOLLOWING: " + count);
      }

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

      Matcher pattern = palletteStart.matcher(line);
      if (pattern.find()) {
        System.out.println("ENTERED THE LINE " + line);
        isInPallette = true;
      }
    }

    return new VoxelModel(width, height, depth, voxels, List.of());
  }
}

