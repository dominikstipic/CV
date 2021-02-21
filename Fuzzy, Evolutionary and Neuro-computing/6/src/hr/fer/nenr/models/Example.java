package hr.fer.nenr.models;

import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class Example{
	public final List<Double> label;
	public final List<Double> inputs;
	
	public Example(List<Double> inputs, List<Double> label) {
		this.label = label;
		this.inputs = inputs;
	}

	public Matrix getInput() {
		Matrix input = MatrixAdapter.toVec(inputs);
		return input;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((inputs == null) ? 0 : inputs.hashCode());
		result = prime * result + ((label == null) ? 0 : label.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Example other = (Example) obj;
		if (inputs == null) {
			if (other.inputs != null)
				return false;
		} else if (!inputs.equals(other.inputs))
			return false;
		if (label == null) {
			if (other.label != null)
				return false;
		} else if (!label.equals(other.label))
			return false;
		return true;
	}

	@Override
	public String toString() {
		String x = "inputs=" + inputs;
		String y = "label=" + label;
		return x + ", " + y;
	}

	

	
}
