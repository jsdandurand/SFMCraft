package me.image2mc.image2MC;

import me.image2mc.image2MC.commands.TestBlockCommand;
import me.image2mc.image2MC.listeners.DevPlayerListener;
import org.bukkit.plugin.java.JavaPlugin;

public final class Image2MC extends JavaPlugin {

  @Override
  public void onEnable() {
    // TODO: Comment this out, or find a better way to handle dev env
    // This method is called to set the joined player to creative mode. Makes testing much easier.
    System.out.println("Image2MC plugin has been enabled! V4");
    getServer().getPluginManager().registerEvents(new DevPlayerListener(), this);
    this.getCommand("testblock").setExecutor(new TestBlockCommand());
  }

  @Override
  public void onDisable() {
    System.out.println("Image2MC plugin has been disabled!");
  }
}
