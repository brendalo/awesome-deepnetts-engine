/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */
    
package deepnetts.util;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class ImageUtils {
        
    /**
     * Scales input image to specified target width or height, centers and returns resulting image.
     * Scaling factor is calculated using larger dimension (width or height).
     * Keeps aspect ratio and image type, and bgColor parameter to fill background. 
     * 
     * @param img
     * @param targetWidth
     * @param targetHeight
     * @param bgColor
     * @return scaled and centered image
     */
    public static BufferedImage scaleAndCenter(BufferedImage img, int targetWidth, int targetHeight, int padding, Color bgColor) {

        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
                
        float scale = 0;
        int xPos, yPos;
        
        if (imgWidth > imgHeight) {
            scale = imgWidth / (float)(targetWidth-2*padding);

        } else { // imgHeight < imgWidth
            scale = imgHeight / (float)(targetHeight-2*padding);

        }

        int newWidth = (int) (imgWidth / scale);
        int newHeight = (int)(imgHeight / scale);
        
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, imgWidth);
        
        BufferedImage resultImg = new BufferedImage(targetWidth, targetHeight, img.getType());
        resultImg.getGraphics().setColor(bgColor);
        resultImg.getGraphics().fillRect(0, 0, targetWidth, targetHeight);
                
        if (imgWidth > imgHeight) {
            xPos = padding;
            yPos = padding + (targetHeight-2*padding - newHeight) / 2;            
        } else {
            xPos = padding + (targetWidth -2*padding - newWidth) / 2;
            yPos = padding;                                    
        }
        
        resultImg.getGraphics().drawImage(scaledImg, xPos, yPos, null);
                    
        return resultImg;
    }

    /**
     * Scales input image to specified target width or height, crops and returns resulting image.
     * Scaling factor is calculated using smaller dimension (width or height).
     * Keeps aspect ratio and image type, and bgColor parameter to fill background. 
     * 
     * @param img image to scale
     * @param targetWidth target image width
     * @param targetHeight target image height
     * @return scaled and cropped image
     */
    public static BufferedImage scaleBySmallerAndCrop(BufferedImage img, int targetWidth, int targetHeight) {

        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
                
        float scale = 0;

        
        if (imgWidth < imgHeight) { // which one is smaller width or height?
            scale = imgWidth / (float)targetWidth; // scale by width
        } else { // imgHeight < imgWidth // scale by height
            scale = imgHeight / (float)targetHeight;
        }

        int newWidth = (int) (imgWidth / scale);
        int newHeight = (int)(imgHeight / scale);
        
        Image scaledImg = img.getScaledInstance(newWidth, newHeight, imgWidth);
               
        BufferedImage scaledBuffImg = new BufferedImage(newWidth, newHeight, img.getType());
        scaledBuffImg.getGraphics().drawImage(scaledImg, 0, 0, null);
        
        BufferedImage resultImage = null;
        
        if (imgWidth < imgHeight) { // crop by centering on  height
            final int xPos = 0;
            final int yPos = (newHeight - targetHeight) / 2;            
            resultImage = scaledBuffImg.getSubimage(xPos, yPos, targetWidth, targetHeight);            
        } else { // crop by centering on height
            final int xPos = (newWidth-targetWidth) / 2;
            final int yPos = 0;                                    
            resultImage = scaledBuffImg.getSubimage(xPos, yPos, targetWidth, targetHeight);            
        }
        
        return resultImage;
    }
    
    
    /**
     * Loads all images from the specified directory, and returns them as a list.
     * 
     * @param dir
     * @return list of images as BufferedImage instances
     * @throws IOException 
     */
    public static List<BufferedImage> loadImagesFromDirectory(File dir) throws IOException {
        List<BufferedImage> imageList = new ArrayList<>();
        for (final File file : dir.listFiles()) {
            if (!file.isDirectory()) {
                BufferedImage img = ImageIO.read(file);
                imageList.add(img);
            } 
        }        
        return imageList;
    }
    
    
    /**
     * Loads images (jpg, jpeg, png) from specificed directory and returns them as a map with File object as a key and BufferedImage object as a value.
     *  
     * @param dir
     * @return
     * @throws IOException 
     */
    public static HashMap<File, BufferedImage> loadFileImageMapFromDirectory(File dir) throws IOException {
        if (!dir.isDirectory()) throw new IllegalArgumentException("Parameter dir must be a directory: "+dir.toString());
        
        HashMap<File, BufferedImage> imageMap = new HashMap<>();
        for (final File file : dir.listFiles()) {
            if (file.isDirectory()) continue;// skip subdirectories
            
            final String imgType = getImageType(file);                       
            if (!imgType.equalsIgnoreCase("jpg") && !imgType.equalsIgnoreCase("jpeg") && !imgType.equalsIgnoreCase("png")) continue;
                                        
            BufferedImage img = ImageIO.read(file);
            imageMap.put(file, img);            
        }        
        return imageMap;
    }    
    
    public static String getImageType(final File file) {
            final String fileName = file.getName();
            return fileName.substring(fileName.lastIndexOf(".")+1); // get file ext/img type                       
    }
    
    
    // hardcoded for lego replace with DataSetUtils
    public static void createIndexFile(HashMap<File, BufferedImage> imageMap, String imageFile, boolean useAbsPath) throws IOException {
        // dodaj regex pomocu koga ce da ih ubacuje uklas?
        try (BufferedWriter out = new BufferedWriter(new FileWriter(imageFile))) {
            int fileCount = imageMap.size();
            int i = 0;

            for (File file : imageMap.keySet()) {