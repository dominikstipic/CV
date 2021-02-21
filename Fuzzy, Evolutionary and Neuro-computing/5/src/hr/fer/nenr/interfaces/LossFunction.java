package hr.fer.nenr.interfaces;

import java.util.List;

public interface LossFunction{
	double loss(List<Double> out, List<Integer> target);
	double loss(List<Double> out, int classId);
	List<Double> backward();

	
	
}
