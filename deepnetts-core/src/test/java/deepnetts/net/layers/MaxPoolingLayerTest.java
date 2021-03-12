package deepnetts.net.layers;

import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class MaxPoolingLayerTest {

    public MaxPoolingLayerTest() {
    }

    /**
     * Test of forward method, of class MaxPoolingLayer.
     */
    @Test
    public void testForwardSingleChannel() {

        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});

        // set biases to zero
        float[] biases = new float[]{0.0f};

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 1);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter;
        convLayer.biases = biases;

        inputLayer.setInput(input);
        convLayer.forward();    // vidi koliki je output i njega onda pooluj
        /* [-0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
                                      0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742, 
                                      0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
                                      0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
                                      0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
                                     -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983]*/

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(3, 3,
                new float[]{0.45139998f, 0.5405f, 0.4957f,
                    0.45f, 0.99470013f, 0.77730006f,
                    0.4385f, 0.29759997f, 0.23410001f});

        /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5  */
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-8f);
    }

    @Test
    public void testForwardMultiChannel() {

        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 2);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});
        convLayer.filters[1] = new Tensor(3, 3,
                new float[]{0.11f, 0.21f, 0.31f,
                    -0.21f, -0.22f, -0.23f,
                    0.31f, 0.31f, 0.31f});
        // set biases to zero
        convLayer.biases = new float[]{0.0f, 0.0f};

        inputLayer.setInput(input);
        convLayer.forward();    // output from convolutional layer:
        /*         
                                    -0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
                                     0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742,
                                     0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
                                     0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
                                     0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
                                    -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983,
                                    
                                    -0.20889999, -0.26760003, -0.010199998, -6.999895E-4,  0.22350001,   0.22450002,
                                     0.3319,      0.48950002,  0.44680002,   0.4791,       0.40600002,   0.25570002, 
                                     0.19569999,  0.3932,      0.2622,       0.014099985, -0.060699996, -0.15130001, 
                                     0.2328,      0.3976,      0.6252,       0.6627,       0.8222,       0.4177, 
                                     0.27,        0.31350002,  0.23630002,  -0.0035999827, 0.04750003,   0.10620001, 
                                     0.028599992, 0.105699986, 0.18150005,   0.033699997,  0.064200014, -0.014600009"        
         */

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(3, 3, 2,
                new float[]{0.45139998f, 0.5405f, 0.4957f,
                    0.45f, 0.99470013f, 0.77730006f,
                    0.4385f, 0.29759997f,