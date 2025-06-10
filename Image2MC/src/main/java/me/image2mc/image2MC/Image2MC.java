package me.image2mc.image2MC;

import org.bukkit.plugin.java.JavaPlugin;

public final class Image2MC extends JavaPlugin {

  @Override
  public void onEnable() {
    System.out.println("Image2MC plugin has been enabled!");
  }

  @Override
  public void onDisable() {
    System.out.println("Image2MC plugin has been disabled!");
  }
}
