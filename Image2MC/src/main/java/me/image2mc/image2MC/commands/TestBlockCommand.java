package me.image2mc.image2MC.commands;

import me.image2mc.image2MC.loaders.VoxelLoader;
import me.image2mc.image2MC.models.Voxel;
import me.image2mc.image2MC.models.VoxelModel;
import me.image2mc.image2MC.parsers.PyVoxelParser;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.util.Vector;

import java.io.File;
import java.util.List;

public class TestBlockCommand implements CommandExecutor {
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

      VoxelLoader loader = new VoxelLoader(List.of(new PyVoxelParser()));
      VoxelModel model = loader.load(voxelFile);

      World world = player.getWorld();
      Vector direction = player.getLocation().getDirection().normalize().multiply(50);
      int originX = player.getLocation().getBlockX() + direction.getBlockX();
      int originY = player.getLocation().getBlockY();
      int originZ = player.getLocation().getBlockZ() + direction.getBlockZ();

      for (Voxel v : model.voxels()) {
        world.getBlockAt(originX + v.x(), originY + v.y(), originZ + v.z())
            .setType(Material.DIAMOND_BLOCK);
      }

      player.sendMessage("Model placed successfully!");

    } catch (Exception e) {
      sender.sendMessage("Failed to load model: " + e.getMessage());
      e.printStackTrace();
    }

    return true;
  }

}