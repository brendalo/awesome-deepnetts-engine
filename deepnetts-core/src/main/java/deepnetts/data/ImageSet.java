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
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
    
package deepnetts.data;

import deepnetts.core.DeepNetts;
import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Represents data set with images
 * 
 * @author zoran
 */
public class ImageSet extends DataSet<ExampleImage> { 
    // ovi ne mogu svi da budu u memoriji odjednom...
    // this should be items
    private final List<ExampleImage> images; // mozda neka konkurentna kolekcija da vise threadova moze paralelno da trenira i testira nekoliko neuronskih mreza
    private final List<String> labels;
    private final int imageWidth;
    private final int imageHeight;
    private Tensor mean;
    
    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());    
        
   // osmisliti i neki protocni / buffered data set, koji ucitava jedan batch
      
    public ImageSet(int imageWidth, int imageHeight) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        
        images = new ArrayList();       
        labels = new ArrayList();     
    }    

    public ImageSet(int imageWidth, int imageHeight, int capacity) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        
        images = new ArrayList(capacity);       
        labels = new ArrayList();     
    }        
    
    /**
     * Adds image to this image set.
     * 
     * @param image
     * @throws DeepNettsException if image is empty or has wrong dimensions.
     */
    public void add(ExampleImage image) throws DeepNettsException {
        if (image == null) throw new DeepNettsException("Example image cannot b