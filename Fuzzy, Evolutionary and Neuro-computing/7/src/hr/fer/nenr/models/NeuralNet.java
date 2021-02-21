package hr.fer.nenr.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

import org.apache.commons.lang3.ArrayUtils;

import hr.fer.nenr.dataset.IDataset;
import hr.fer.nenr.utils.LinAlg;
import hr.fer.nenr.utils.Utils;

public class NeuralNet {
	public final Function<Double, Double> SIGM = x -> 1/(1+Math.exp(x));
	public final int in;
	public final int out;
	public final int depth;
	public final int neuronNum;
	public final int[] dimensions;
	public final int paramSize;
	
	private double[][] scaleParam;
	private double[][] groupParam;
	private List<Double[][]> weights = new ArrayList<>();
	private List<Double[]> biases = new ArrayList<>();
	private double[] queue;
	
	public NeuralNet(int ...dim) {
		dimensions = dim;
		in = dim[0];
		out = dim[dim.length-1];
		depth = dim.length - 1;
		neuronNum = LinAlg.sum(dimensions) - in;
		queue = new double[neuronNum];
		initParams(dim);
		paramSize = paramSizeList().stream().reduce(0, (a,b) -> a+b);
	}
	
	private void initParams(int[] dim) {
		groupParam = new double[dim[1]][in];
		scaleParam = new double[dim[1]][in];
		for(int i = 1; i < dim.length-1; ++i) {
			int inDim = dim[i];
			int outDim = dim[i+1];
			Double [][]W = new Double[outDim][inDim];
			Double []B = new Double[outDim];
			weights.add(W);
			biases.add(B);
		}
	}
	
	public List<Integer> paramSizeList() {
		List<Integer> dims = new LinkedList<>();
		dims.add(in*dimensions[1]);
		dims.add(in*dimensions[1]);
		for(int i = 0; i < weights.size(); ++i) {
			Double[][] W = weights.get(i);
			Double[] b = biases.get(i);
			dims.add(W.length*W[0].length);
			dims.add(b.length);
		}
		return dims;	
	}
	
	private double getLoss(double[] input, double[] target, double[] params) {
		double[] output = calcOutput(input, params);
		double[] diff = LinAlg.minus(output, target);
		diff = LinAlg.pow(diff, 2);
		double loss = LinAlg.sum(diff);
		return loss;
	}
	
	public double calcError(IDataset<Data> dataset, double[] params) {
		setParams(params);
		int N = dataset.size();
		double error = 0;
		for(Data data : dataset) {
			double loss = getLoss(data.example, data.oneHot, params);
			error += loss;
		}
		error /= N;
		return error;
	}
	
	public double[] calcOutput(double[] input, double[] params) {
		setParams(params);
		type1LayerForward(input);
		for(int i = 0; i < depth-2; ++i) {
			Double [][] W = weights.get(i);
			Double [] B = biases.get(i);
			type2LayerForward(W,B);
		}
		double output[] = Arrays.copyOf(queue, out);
		return output;
	}
	
	public double[] predict(double[] input) {
		type1LayerForward(input);
		for(int i = 0; i < depth-2; ++i) {
			Double [][] W = weights.get(i);
			Double [] B = biases.get(i);
			type2LayerForward(W,B);
		}
		double output[] = Arrays.copyOf(queue, out);
		for(int i = 0; i < output.length; ++i) {
			output[i] = output[i] >= 0.5 ? 1 : 0;
		}
		output = Arrays.copyOfRange(output, 0, out);
		return output;
	}
	
	public double[] forwardLayer(int k, double[] input) {
		clearQueue();
		if(k == 0) {
			type1LayerForward(input);
		}
		else {
			Double [][] W = weights.get(k-1);
			Double [] B = biases.get(k-1);
			type2LayerForward(W,B);
		}
		double[] out = Arrays.copyOfRange(queue, 0, dimensions[k+1]);
		return out;
	}
	
	private void type1LayerForward(double[] input) {
		int neuronNum = groupParam.length;
		for(int i = 0; i < neuronNum; ++i) {
			double[] neuronWeight = groupParam[i];
			double[] neuronScale  = LinAlg.abs(scaleParam[i]);
			double[] xs = LinAlg.minus(input, neuronWeight);
			xs = LinAlg.abs(xs);
			xs = LinAlg.div(xs, neuronScale);
			queue[i] = 1/(1+LinAlg.sum(xs));
		}
	}
	
	private void type2LayerForward(Double[][] W, Double[] b) {
		int neuronNum = W.length;
		int inputLen = W[0].length;
		double[] inputs = Arrays.copyOf(queue, inputLen);
		for(int i = 0; i < neuronNum; ++i) {
			double[] neuronWeights = org.apache.commons.lang3.ArrayUtils.toPrimitive(W[i]);
			double logit = LinAlg.scalarMul(neuronWeights, inputs) + b[i]; 
			queue[i] = SIGM.apply(logit);
		}
	}
	
	public List<Double[][]> getLayerParams(int layerId){
		List <Double[][]> params = new ArrayList<>();
		if(layerId == 0) {
			Double[][] G = toObject(groupParam);
			Double[][] S = toObject(scaleParam);
			params.addAll(Arrays.asList(G,S));
		}
		else {
			Double[][] W = weights.get(layerId-1);
			Double[] b = biases.get(layerId-1);
			Double[][] B = new Double[b.length][1];
			for(int i = 0; i < b.length; ++i)B[i][0] = b[i];
			params.addAll(Arrays.asList(W,B));
		}
		return params;
	} 
	
	private Double[][] toObject(double[][] m){
		int rows = m.length;
		int cols = m[0].length;
		Double[][] M = new Double[rows][cols];
		for(int i = 0; i < rows; ++i) {
			M[i] = org.apache.commons.lang3.ArrayUtils.toObject(m[i]);
		}
		return M;
	}
	
	public void setParams(double[] params) {
		// params : |Weight|Scale|W1|B1|W2|B2|W3|B3|
		if(params.length != paramSize) throw new IllegalArgumentException("This parameters can't be used in this NN");
		
		int layer1Neurons = dimensions[1];
		int layer1ParamsNum  = 2*layer1Neurons * in;
		
		double[] groupParams = Arrays.copyOf(params, layer1ParamsNum/2);
		double[] scaleParams = Arrays.copyOfRange(params, layer1ParamsNum/2, layer1ParamsNum);
		
		double[][] group = Utils.toMatrix(groupParams, layer1Neurons, in);
		double[][] scale = Utils.toMatrix(scaleParams, layer1Neurons, in);
		
		List<Double[][]> ws = new ArrayList<>();
		List<Double[]> bs   = new ArrayList<>();
		double[] neruon2Params = Arrays.copyOfRange(params, layer1ParamsNum, params.length);
		int paramPointer = 0;
		for(int i = 1; i < depth; ++i) {
			int out = dimensions[i+1];
			int in  = dimensions[i];
			int numOfParams = out*in + out;
			double[] layerParams = Arrays.copyOfRange(neruon2Params, paramPointer, paramPointer + numOfParams);
			paramPointer += numOfParams;
			double[] W_serialized = Arrays.copyOfRange(layerParams, 0, out*in);
			double[] b = Arrays.copyOfRange(layerParams, out*in, numOfParams);
			Double[][] W = toObject(Utils.toMatrix(W_serialized, out, in));
			Double[] B = org.apache.commons.lang3.ArrayUtils.toObject(b);
			ws.add(W);bs.add(B);
		}
		this.groupParam = group;
		this.scaleParam = scale;
		this.weights = ws;
		this.biases  = bs; 
	}
	
	public void setParams(Double[] params) {
		double[] paramPrimitive = org.apache.commons.lang3.ArrayUtils.toPrimitive(params);
		setParams(paramPrimitive);
	}
	
	public void printParams() {
		System.out.println("Group:");
		Utils.printMatrix(groupParam);
		System.out.println("---");
		System.out.println("Scale");
		Utils.printMatrix(scaleParam);
		System.out.println("---");
		for(int i = 0; i < weights.size(); ++i) {
			System.out.println("W" + i + ":");
			Utils.printMatrix(weights.get(i));
			
			System.out.println("b" + i + ":");
			System.out.println(Arrays.toString(biases.get(i)));
			System.out.println("---");
		}
	}
	
	
	public double[][] getScaleParam() {
		return scaleParam;
	}

	public double[][] getGroupParam() {
		return groupParam;
	}

	public List<Double[][]> getWeights() {
		return weights;
	}

	public List<Double[]> getBiases() {
		return biases;
	}
	
	public double[][] asArray() {
		double[][] G = getGroupParam();
		double[][] S = getScaleParam();
		double[] type1Layer = ArrayUtils.addAll(LinAlg.vectorize(G), LinAlg.vectorize(S));
		
		double[] type2Layers = null;
		for(Double[][] W : weights) {
			double[] w_serial = LinAlg.vectorize(W);
			type2Layers = ArrayUtils.addAll(type2Layers, w_serial);
		}
		for(Double[] B : biases) {
			double[] b = ArrayUtils.toPrimitive(B);
			type2Layers = ArrayUtils.addAll(type2Layers, b);
		}
		
		double[][] result = {type1Layer, type2Layers};
		return result;
	}
	
	public double[] asArray(int layerId) {
		List<Double[][]> params = getLayerParams(layerId);
		double[] vector = null;
		for(Double[][] param : params) {
			double[] vec = LinAlg.vectorize(param);
			vector = ArrayUtils.addAll(vector, vec);
		}
		return vector;
	}

	public NeuralNet copy() {
		NeuralNet net = new NeuralNet(dimensions);
		return net;
	}
	
	private void clearQueue() {
		queue = LinAlg.rep(0., queue.length);
	}
	
}
