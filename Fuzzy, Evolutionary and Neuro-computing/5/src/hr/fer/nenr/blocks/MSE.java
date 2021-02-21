package hr.fer.nenr.blocks;

import java.util.List;
import java.util.stream.Collectors;

import hr.fer.nenr.utils.VectorUtils;

public class MSE extends Loss{

	@Override
	public double loss(List<Double> out, List<Integer> target) {
		VectorUtils.dimCheck(out, target);
		List<Double> doubleTarget = target.stream().map(i -> (double) i).collect(Collectors.toList());
		List<Double> residuls = VectorUtils.minus(doubleTarget, out);
		List<Double> squared = VectorUtils.power(residuls, 2);
		double result = VectorUtils.sum(squared).get(0);
		return result;
	}
	
	
}
