package hr.fer.nenr.loss;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import hr.fer.nenr.utils.VectorUtils;

public class MSE extends Loss{
	private List<Double> output;
	private List<Double> target;

	private void saveInputs(List<Double> output, List<Double> target) {
		this.output = new ArrayList<>(output);
		this.target = new ArrayList<>(target);
	}
	
	@Override
	public double loss(List<Double> out, List<Integer> target) {
		VectorUtils.dimCheck(out, target);
		List<Double> doubleTarget = target.stream().map(i -> (double) i).collect(Collectors.toList());
		saveInputs(out, doubleTarget);
		List<Double> residuls = VectorUtils.minus(doubleTarget, out);
		List<Double> squared = VectorUtils.power(residuls, 2);
		double result = VectorUtils.sum(squared).get(0);
		return 0.5*result;
	}

	@Override
	public List<Double> backward() {
		List<Double> results = VectorUtils.minus(target, output);
		return results;
	}

	
	
	
}
