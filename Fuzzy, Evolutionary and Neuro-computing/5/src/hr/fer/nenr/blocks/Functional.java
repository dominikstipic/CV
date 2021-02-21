package hr.fer.nenr.blocks;

import Jama.Matrix;
import hr.fer.nenr.models.Parameters;

public abstract class Functional extends Component{

	@Override
	public boolean hasParam() {
		return false;
	}
	
	@Override
	public void cleanGradients() {
		throw new UnsupportedOperationException("Component doesn't have gradients");
	}

	@Override
	public Parameters getParams() {
		throw new UnsupportedOperationException("Component doesn't have gradients");
	}

	@Override
	public void setParams(Parameters params) {
		throw new UnsupportedOperationException("Component doesn't have gradients");
	}

	@Override
	public Matrix[] getGrads() {
		throw new UnsupportedOperationException("Component doesn't have gradients");
	}

	
	
	
}
