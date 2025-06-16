package me.image2mc.image2MC.parsers;

import me.image2mc.image2MC.models.Pallette;
import me.image2mc.image2MC.models.Voxel;
import me.image2mc.image2MC.models.VoxelModel;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.*;

public class NPyVoxelParser implements VoxelParser {

    @Override
    public boolean supports(File file) {
        return file.getName().endsWith(".npy");
    }


    // Based on the .npy file format specification here: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
    // this is not a complete parser, it will only work with the specific format we expect from our voxel data.
    // Which is a HxWxLx4 float array where the first float is presence (0-1) and the next three are RGB values (0-1).
    @Override
    public VoxelModel parse(File file) throws IOException {
        try (FileInputStream fis = new FileInputStream(file)) {
            // Seek for MAGIC bytes "NUMPY" with a space before
            byte[] magic = fis.readNBytes(6);
            if (!Arrays.equals(magic, new byte[]{(byte) 0x93, 'N', 'U', 'M', 'P', 'Y'})) {
                throw new IOException("Invalid .npy file format");
            }

            // Version check, we only support version 1.0 for now
            int major = fis.read();
            int minor = fis.read();
            if (major != 1 || minor != 0) {
                throw new IOException("Unsupported .npy version: " + major + "." + minor);
            }

            // Header is preceded by a 2-byte length field
            byte[] headerLenBytes = fis.readNBytes(2);
            int headerLen = ByteBuffer.wrap(headerLenBytes).order(ByteOrder.LITTLE_ENDIAN).getShort() & 0xFFFF;
            byte[] headerBytes = fis.readNBytes(headerLen);
            String header = new String(headerBytes, StandardCharsets.US_ASCII).trim();

            Pattern shapePattern = Pattern.compile("'shape': \\((\\d+), (\\d+), (\\d+), (\\d+)\\)");
            Matcher matcher = shapePattern.matcher(header);
            if (!matcher.find()) throw new IOException("Could not extract shape from header.");

            int h = Integer.parseInt(matcher.group(1));
            int w = Integer.parseInt(matcher.group(2));
            int l = Integer.parseInt(matcher.group(3));
            int c = Integer.parseInt(matcher.group(4));
            if (c != 4) throw new IOException("Expected shape HxWxLx4, got last dim = " + c);

            // Read the actual data
            int totalFloats = h * w * l * c;
            byte[] raw = new byte[totalFloats * 4];
            int bytesRead = 0;
            while (bytesRead < raw.length) {
                int read = fis.read(raw, bytesRead, raw.length - bytesRead);
                if (read == -1) throw new IOException("Unexpected EOF");
                bytesRead += read;
            }

            FloatBuffer floatBuffer = ByteBuffer.wrap(raw)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();

            Map<String, Integer> paletteMap = new LinkedHashMap<>();
            List<Pallette> paletteList = new ArrayList<>();
            List<Voxel> voxelList = new ArrayList<>();

            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    for (int k = 0; k < l; k++) {
                        float presence = floatBuffer.get();
                        float r = floatBuffer.get();
                        float g = floatBuffer.get();
                        float b = floatBuffer.get();

                        if (presence < 0.5f) continue;

                        int red = Math.max(0, Math.min(255, Math.round(r * 255)));
                        int green = Math.max(0, Math.min(255, Math.round(g * 255)));
                        int blue = Math.max(0, Math.min(255, Math.round(b * 255)));

                        String key = red + "," + green + "," + blue;
                        int paletteIndex = paletteMap.computeIfAbsent(key, k2 -> {
                            paletteList.add(new Pallette(red, green, blue));
                            return paletteList.size() - 1;
                        });
                        // y, and z axis needs to be inverted cause it's flipped in MC
                        voxelList.add(new Voxel(i, h - 1 - j, l - 1 - k, paletteIndex));
                    }
                }
            }

            return new VoxelModel(w, h, l, voxelList, paletteList);
        }
    }
}