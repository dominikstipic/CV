package hr.fer.nenr.blocks;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import hr.fer.nenr.utils.VectorUtils;

public class Sigmoid extends Functional{
	private List<Double> memory;
	private Function<Double, Double> function = d -> 1/(1+Math.exp(-d));
	
	public Sigmoid(int in) {
		this.in = in;
		this.out = in;
	}

	@Override
	public List<Double> forward(List<Double> inputs) {
		VectorUtils.dimCheck(inputs, in);
		memory = new ArrayList<>(inputs);
		inputs = VectorUtils.vectorUniFunction(inputs, function);
		return inputs;
	}

	@Override
	public List<Double> backward(List<Double> gradients) {
		VectorUtils.dimCheck(gradients, memory);
		List<Double> outputs = VectorUtils.vectorUniFunction(memory, function);
		List<Double> inputGrads = outputs.stream().map(d -> d*(1-d)).collect(Collectors.toList());
		inputGrads = VectorUtils.mul(inputGrads, gradients);
		return inputGrads;
	}


	
}
