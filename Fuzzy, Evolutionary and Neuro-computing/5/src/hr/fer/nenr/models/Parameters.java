package hr.fer.nenr.models;

import Jama.Matrix;
import hr.fer.nenr.utils.MatrixAdapter;

public class Parameters {
	public final Matrix W;
	public final Matrix b;
	
	public Parameters(Matrix w, Matrix b) {
		this.W = w;
		this.b = b;
	}

	@Override
	public String toString() {
		String ws = MatrixAdapter.matrixString(W);
		String bs = MatrixAdapter.matrixString(b);
		String s = String.format("WEIGHTS:\n%s\nBIAS: %s", ws,bs);
		return s;
	}
	
}
