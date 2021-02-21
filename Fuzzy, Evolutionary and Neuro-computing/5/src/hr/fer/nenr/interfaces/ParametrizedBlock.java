package hr.fer.nenr.interfaces;

import java.util.List;

public interface ParametrizedBlock extends ComputationalBlock{
	void setParameters(List<Double> params);
}
