package hr.fer.nenr.blocks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Supplier;

import hr.fer.nenr.interfaces.ComputationalBlock;
import hr.fer.nenr.nn.Initialization;
import hr.fer.nenr.utils.VectorUtils;

public class Summator implements ComputationalBlock{
	private List<Double> memory;
	private List<Double> weights;
	private int in;
	private List<Double> grads;
	
	public Summator(int in) {
		this.in = in;
		cleanGradients();
		initWeights(Initialization.uniform(-1, 1));
	}
	
	public void initWeights(Supplier<Double> supplier) {
		weights = new ArrayList<>();
		for(int i = 0; i < in+1; ++i) {
			double value = supplier.get();
			weights.add(value);
		}
	}
	
	private List<Double> processInput(List<Double> input){
		List<Double> arr = new ArrayList<>(input);
		arr.add(0, 1.0);
		return arr;
	}
	
	@Override
	public List<Double> forward(List<Double> inputs) {
		VectorUtils.dimCheck(inputs, in);
		inputs = processInput(inputs);
		memory = new ArrayList<>(inputs);
		inputs = VectorUtils.mul(weights, inputs);
		inputs = VectorUtils.sum(inputs);
		return inputs;
	}

	private void accumulateGrads(List<Double> gradients) {
		grads = VectorUtils.add(grads, gradients);
	}
	
	@Override
	public List<Double> backward(List<Double> gradients) {
		VectorUtils.dimCheck(gradients, 1);
		List<Double> dw = VectorUtils.scalarMul(memory, gradients.get(0));
		List<Double> dx = VectorUtils.scalarMul(weights, gradients.get(0));
		accumulateGrads(dw);
		return dx;
	}

	@Override
	public int inputFeatures() {
		return in;
	}

	@Override
	public int outFeatures() {
		return 1;
	}

	@Override
	public void cleanGradients() {
		grads = VectorUtils.replicate(0.0, in+1);
	}

	@Override
	public Object getParams() {
		return weights;
	}

	@Override
	public void setParams(Object params) {
		List<Double> paramList = (List<Double>) params;
		weights = paramList;
	}

	@Override
	public boolean hasParam() {
		return true;
	}

	@Override
	public Object getGrads() {
		return grads;
	}

	
	

}
