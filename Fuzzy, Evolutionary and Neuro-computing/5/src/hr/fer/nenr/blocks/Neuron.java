package hr.fer.nenr.blocks;

import java.util.List;

import hr.fer.nenr.interfaces.ComputationalBlock;

public class Neuron implements ComputationalBlock{
	private int in;
	private Summator summator;
	private Sigmoid sigmoid;
	
	public Neuron(int in) {
		this.in = in;
		summator = new Summator(in);
		sigmoid = new Sigmoid(1);
	}

	@Override
	public List<Double> forward(List<Double> inputs) {
		inputs = summator.forward(inputs);
		inputs = sigmoid.forward(inputs);
		return inputs;
	}

	@Override
	public List<Double> backward(List<Double> gradients) {
		gradients = sigmoid.backward(gradients);
		gradients = summator.backward(gradients);
		return gradients;
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
		summator.cleanGradients();
	}

	@Override
	public Object getParams() {
		return summator.getParams();
	}

	@Override
	public void setParams(Object params) {
		List<Double> vector = (List<Double>) params;
		summator.setParams(vector);
	}

	@Override
	public boolean hasParam() {
		return true;
	}

	@Override
	public Object getGrads() {
		List<Double> grads = (List<Double>) summator.getGrads();
		return grads;
	}
	
	
	
	
}
