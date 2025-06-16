package me.image2mc.image2MC.commands;

import me.image2mc.image2MC.loaders.VoxelLoader;
import me.image2mc.image2MC.models.Pallette;
import me.image2mc.image2MC.models.Voxel;
import me.image2mc.image2MC.models.VoxelModel;
import me.image2mc.image2MC.parsers.NPyVoxelParser;
import me.image2mc.image2MC.parsers.PyVoxelParser;
import me.image2mc.image2MC.texture.TextureColors;
import org.bukkit.DyeColor;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.util.Vector;

import java.io.File;
import java.util.List;
import java.util.Map;

public class TestBlockCommand implements CommandExecutor {

  // THANK YOU! @DMan16 https://www.spigotmc.org/threads/easiest-way-of-comparing-rgb-values-to-a-minecraft-block.513312/
  public Material woolFromColor(int red, int green, int blue) {
    int distance = Integer.MAX_VALUE;
    DyeColor closest = null;
    for (org.bukkit.DyeColor dye : org.bukkit.DyeColor.values()) {
      org.bukkit.Color color = dye.getColor();
      // Add a small amount of random noise to each RGB value
      int tempRed = color.getRed() + (int) (Math.random() * 3) - 1;
      int tempGreen = color.getGreen() + (int) (Math.random() * 3) - 1;
      int tempBlue = color.getBlue() + (int) (Math.random() * 3) - 1;
      int dist = Math.abs(color.getRed() - red) + Math.abs(color.getGreen() - green) + Math.abs(color.getBlue() - blue);
      if (dist < distance) {
        distance = dist;
        closest = dye;
      }
    }
    // You might want to add a try here - I'm not sure how it worked back in 1.14 (it may produce a NullPointerException)
    return Material.getMaterial((closest.name() + "_wool").toUpperCase());
  }

  public Material closestTexture(int red, int green, int blue) {
    int minDistance = Integer.MAX_VALUE;
    Material closest = null;

    for (Map.Entry<Material, Pallette> entry : TextureColors.colorMap.entrySet()) {
      Pallette p = entry.getValue();
      int dist = Math.abs(p.red() - red) + Math.abs(p.green() - green) + Math.abs(p.blue() - blue);
      if (dist < minDistance) {
        minDistance = dist;
        closest = entry.getKey();
      }
    }

    return closest;
  }

  @Override
  public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
    if (!(sender instanceof Player player)) {
      sender.sendMessage("This command can only be run by a player.");
      return true;
    }

    if (args.length < 1) {
      player.sendMessage("Usage: /testblock <path_to_voxel_py_file>");
      return true;
    }

    try {
      // Combine args into a file path (in case the path has spaces)
      String path = String.join(" ", args);
      // Remove quotes if present
      path = path.replace("\"", "").replace("'", "");
      File voxelFile = new File(path);

      if (!voxelFile.exists()) {
        player.sendMessage("File does not exist: " + path);
        return true;
      }

      VoxelLoader loader = new VoxelLoader(List.of(new PyVoxelParser(), new NPyVoxelParser()));
      VoxelModel model = loader.load(voxelFile);

      World world = player.getWorld();
      Vector direction = player.getLocation().getDirection().normalize().multiply(50);
      int originX = player.getLocation().getBlockX() + direction.getBlockX();
      int originY = player.getLocation().getBlockY();
      int originZ = player.getLocation().getBlockZ() + direction.getBlockZ();

      for (Voxel v : model.voxels()) {
        Pallette modelPallette = model.palletes().get(v.color());
        world.getBlockAt(originX + v.x(), originY + v.y(), originZ + v.z())
            .setType(this.closestTexture(modelPallette.red(), modelPallette.green(), modelPallette.blue()));
      }

      player.sendMessage("Model placed successfully!");

    } catch (Exception e) {
      sender.sendMessage("Failed to load model: " + e.getMessage());
      e.printStackTrace();
    }

    return true;
  }

}