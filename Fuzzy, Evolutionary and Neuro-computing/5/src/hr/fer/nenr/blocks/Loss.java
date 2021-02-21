package hr.fer.nenr.blocks;

import java.util.List;

import hr.fer.nenr.interfaces.LossFunction;
import hr.fer.nenr.utils.VectorUtils;

public abstract class Loss implements LossFunction{

	@Override
	public abstract double loss(List<Double> out, List<Integer> target);

	@Override
	public double loss(List<Double> out, int classId) {
		if(out.size() <= classId) throw new IllegalArgumentException("classId out of bounds");
		List<Integer> target = VectorUtils.oneHot(classId, out.size());
		return loss(out, target);
	}

}
