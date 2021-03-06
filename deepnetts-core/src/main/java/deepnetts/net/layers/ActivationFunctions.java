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
    
package deepnetts.net.layers;

import deepnetts.util.DeepNettsException;

/**
 * Typical mathematic functions used as layer activation functions in neural networks.
 * 
 * TODO: add slope and amplitude for sigmoid, tanh etc.
 * annottations o automaticly generate enums for activation types?
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public final class ActivationFunctions {
       
    /**
     * Private constructor to prevent instantiation of this class (only static methods)
     */
    private  ActivationFunctions() {  }
        
    
    /**
     * Returns the result of the specified function for specified input.
     * 
     * @param type
     * @param x
     * @return 
     */
    public static final float calc(final ActivationType type, final float x) {

        switch(type) {
            case SIGMOID:
                 return sigmoid(x);
                
            case TANH:
                return tanh(x);
                
            case RELU:
                return relu(x);    
                
            case LINEAR:
                return linear(x);
        }
        
        throw new DeepNettsException("Unknown transfer function type!");
    