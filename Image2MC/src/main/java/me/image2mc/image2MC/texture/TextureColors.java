package me.image2mc.image2MC.texture;

import me.image2mc.image2MC.models.Pallette;
import org.bukkit.Material;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class TextureColors {
  public static final Map<Material, Pallette> colorMap = new HashMap<>(); // Stores block_name --> Pallette

  static {
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(
        TextureColors.class.getResourceAsStream("/average.txt")))) {
      String line;
      while ((line = reader.readLine()) != null) {
        String[] parts = line.split("\t");
        if (parts.length < 3) continue;

        String textureName = parts[0].replace(".png", "").toUpperCase();
        Material material = Material.getMaterial(textureName);
        if (material == null) {
          continue; // Skip if material is not found
        }

        if(material.hasGravity() || !material.isBlock() || !material.isSolid() || material.name().contains("CORAL")
          || !material.isOccluding() || material.isInteractable()){
          continue;
        }

        String[] rgb = parts[2].split(",");
        if (rgb.length < 3) continue;
        int r = Integer.parseInt(rgb[0].trim());
        int g = Integer.parseInt(rgb[1].trim());
        int b = Integer.parseInt(rgb[2].trim());
        colorMap.put(material, new Pallette(r, g, b));
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
