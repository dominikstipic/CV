package nenr.test;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import hr.fer.nenr.models.NeuralNet;
import hr.fer.nenr.utils.LinAlg;

public class NNTest {
	
	public void inputOutputTest() {
		NeuralNet net = new NeuralNet(2,3,4,3);
		assertEquals(net.in, 2);
		assertEquals(net.out, 3);
	}
	
	private void checkDim(Double[][] W1, int n, int m) {
		assertEquals(W1.length, n);
		assertEquals(W1[0].length, m);
	}
	
	public void dimensionTest() {
		NeuralNet net = new NeuralNet(4,2,3,7,10,3);
		double[][] W1 = net.getGroupParam();
		double [][] S = net.getScaleParam();
		assertEquals(W1.length, 2);
		assertEquals(W1[0].length, 4);
		assertEquals(S.length, 2);
		assertEquals(S[0].length, 4);
		
		List<Double[][]> weights = net.getWeights();
		List<Double[]> biases    = net.getBiases();
		
		checkDim(weights.get(0), 3, 2);
		checkDim(weights.get(1), 7, 3);
		checkDim(weights.get(2), 10, 7);
		checkDim(weights.get(3), 3, 10);
		
		assertEquals(biases.get(0).length, 3);
		assertEquals(biases.get(1).length, 7);
		assertEquals(biases.get(2).length, 10);
		assertEquals(biases.get(3).length, 3);
	}
	
	public void neuronNumber() {
		NeuralNet net = new NeuralNet(4,2,3,7,10,3);
		assertEquals(net.neuronNum, 25);
	}
	
	public void paramSize() {
		NeuralNet net = new NeuralNet(4,2,3,7,10,3);
		assertEquals(net.paramSize, 166);
	}
	
	public void setParam1() {
		NeuralNet net = new NeuralNet(2,3);
		double[] params = new double[] {1,2,3,4,5,6, 7,8,9,10,11,12,};
		net.setParams(params);
	}
	
	public void setParam2() {
		NeuralNet net = new NeuralNet(2,3,4,3);
		double[] L1 = new double[] {1,2,3,4,5,6, 7,8,9,10,11,12};
		double[] L2 = LinAlg.rep(1., 12+4);
		double[] L3 = LinAlg.rep(2., 12+3);
		double[] params = org.apache.commons.lang3.ArrayUtils.addAll(L1,L2);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L3);
		net.setParams(params);
		net.printParams();
	}
	
	public void setParam3() {
		NeuralNet net = new NeuralNet(4,2,3,7,10,3);
		double[] L1 = LinAlg.rep(5., 2*4*2);
		double[] L2 = LinAlg.rep(1., 3*2 + 3);
		double[] L3 = LinAlg.rep(2., 3*7 + 7);
		double[] L4 = LinAlg.rep(3., 10*7 + 10);
		double[] L5 = LinAlg.rep(4., 3*10 + 3);
		double[] params = org.apache.commons.lang3.ArrayUtils.addAll(L1,L2);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L3);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L4);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L5);
		net.setParams(params);
	}
	
	public void paramSizeList() {
		NeuralNet net1 = new NeuralNet(2,4,3,2);
		NeuralNet net2 = new NeuralNet(4,2,3,7,10,3);
		int sum1 = net1.paramSizeList().stream().reduce(0, (a,b)->a+b);
		int sum2 = net2.paramSizeList().stream().reduce(0, (a,b)->a+b);
		assertEquals(sum1, net1.paramSize);
		assertEquals(sum2, net2.paramSize);
	}
	
	private NeuralNet getNet() {
		NeuralNet net = new NeuralNet(2,3,4,3);
		double[] L11 = LinAlg.rep(0., 2*3);
		double[] L12 = LinAlg.rep(1., 2*3);
		double[] L2 = LinAlg.rep(2., 12+4);
		double[] L3 = LinAlg.rep(3., 12+3);
		double[] params = org.apache.commons.lang3.ArrayUtils.addAll(L11,L12);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L2);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L3);
		net.setParams(params);
		return net;
	}
	
	public void getNetData() {
		NeuralNet net = getNet();
		assertEquals(net.in, 2);
		assertEquals(net.out, 3);
		assertEquals(net.depth, 3);
		assertEquals(net.paramSize, 43);
	}
	
	public void forwardTest() {
		NeuralNet net = getNet();
		double[] input = {0.3,0.5};
		net.printParams();
		for(int i = 0; i < net.depth; ++i) {
			input = net.forwardLayer(i, input);
			System.out.println(Arrays.toString(input));
		}
	}
	
	private void printMatrix(Double[][] d) {
		for(int i = 0; i < d.length; ++i) {
			for(int j = 0; j < d[0].length; ++j) {
				System.out.print(d[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	@Test
	public void getLayerParams() {
		NeuralNet net = getNet();
		net.printParams();
		for(int i = 0; i < net.depth; ++i) {
			List<Double[][]> list = net.getLayerParams(i);
			printMatrix(list.get(0));
			printMatrix(list.get(1));
			System.out.println("****");
		}
	}
	
}
