package hr.fer.nenr.models;

import java.util.List;

public class Example{
	public final int label;
	public final List<Double> inputs;
	
	public Example(List<Double> inputs, int label) {
		this.label = label;
		this.inputs = inputs;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((inputs == null) ? 0 : inputs.hashCode());
		result = prime * result + label;
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
		if (label != other.label)
			return false;
		return true;
	}

	
}
