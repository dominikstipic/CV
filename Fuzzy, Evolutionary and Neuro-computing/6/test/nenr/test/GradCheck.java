package nenr.test;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import Jama.Matrix;
import hr.fer.nenr.blocks.ComputationalBlock;
import hr.fer.nenr.blocks.FuzzyLayer;
import hr.fer.nenr.blocks.Linear;
import hr.fer.nenr.blocks.NormalizationLayer;
import hr.fer.nenr.blocks.ProductWindow;
import hr.fer.nenr.blocks.Summer;
import hr.fer.nenr.loss.LossFunction;
import hr.fer.nenr.loss.MSE;
import hr.fer.nenr.models.Parameters;
import hr.fer.nenr.utils.MatrixAdapter;

public class GradCheck {
	public final static double epsilon = 10E-6;

	private Matrix shift(Matrix input, boolean isPositive) {
		int N = input.getColumnDimension();
		Matrix e = MatrixAdapter.replicate(epsilon, 1, N);
		e = e.times(0.5);
		if(isPositive) {
			return input.plus(e);
		}
		else {
			return input.minus(e);
		}
	}
	
	private void check(ComputationalBlock block) {
		int D = 3;
		Matrix input = MatrixAdapter.symetricUniform(1, 1, D);
		System.out.println("INPUT " + MatrixAdapter.matrixString(input));
		Matrix e = MatrixAdapter.replicate(1.0, 1, D);
		
		Matrix inPlus = shift(input, true);
		Matrix inMinus = shift(input, false);
		Matrix outPlus = block.forward(inPlus);
		Matrix outMinus = block.forward(inMinus);
		
		Matrix gradApprox = outPlus.minus(outMinus);
		gradApprox = gradApprox.times(1/epsilon);
		String approx = MatrixAdapter.matrixString(gradApprox);
		
		block.forward(input);
		Matrix realGrad = block.backward(e);
		String real = MatrixAdapter.matrixString(realGrad);
		
		System.out.println("REAL GRAD: " + real);
		System.out.println("APPROX GRAD: " + approx);
		
	}
	
	
	private Parameters getParams2() {
		List<String> names = Arrays.asList("W", "b");
		//W = [[p, q],[p, q]]
		Matrix w1 = MatrixAdapter.toVec(.2, -.2, .1);
		Matrix w2 = MatrixAdapter.toVec(.4, -.3, 0.4);
		Matrix W = MatrixAdapter.stack(w1, w2, false);
		Matrix b = MatrixAdapter.toVec(-.3, -.5);
		Parameters p = new Parameters(names, Arrays.asList(W,b));
		return p;
	}
	
	private void test(ComputationalBlock block) {
		Matrix input = MatrixAdapter.toVec(1.0, -2.0, 0.5);
		
		Matrix out = block.forward(input);
		//Matrix e = MatrixAdapter.replicate(1.0, 1, out.getColumnDimension());
		Matrix e = MatrixAdapter.toVec(4., -0.2);
		Matrix g = block.backward(e);

		String inStr = MatrixAdapter.matrixString(input);
		String outStr = MatrixAdapter.matrixString(out);
		String gradStr = MatrixAdapter.matrixString(g);
		
		System.out.println("INPUT: " + inStr);
		System.out.println("OUT: " + outStr);
		System.out.println("GRAD: " + gradStr);
	}
	
	@Test
	public void linTest() {
		Linear lin = new Linear(3, 2);
		lin.setParams(Arrays.asList(getParams2()));
		test(lin);
	}
	
	public void sumTest() {
		Summer s = new Summer(2);
		test(s);
	}
	
	public void normTest() {
		NormalizationLayer s = new NormalizationLayer(3);
		test(s);
	}
	
	public void prodTest() {
		ProductWindow s = new ProductWindow(6, 3);
		test(s);
	}
	
	private Parameters getParams1() {
		List<String> names = Arrays.asList("A", "B");
		Matrix a = MatrixAdapter.toVec(0.1, -.3, 0.2, .7);
		Matrix b = MatrixAdapter.toVec(.2, .4, .5, -.1);
		Parameters p = new Parameters(names, Arrays.asList(a,b));
		return p;
	}
	
	public void fuzzyLayer() {
		FuzzyLayer layer = new FuzzyLayer(2, 4);
		layer.setParams(List.of(getParams1()));
		test(layer);
	}
	
	public void losstest() {
		LossFunction mse = new MSE();
		Matrix input = MatrixAdapter.toVec(1.0, -2.0, 0.5);
		Matrix target = MatrixAdapter.toVec(3, .1, .8);
		
		double out = mse.loss(input, target);
		Matrix g = mse.backward();

		String inStr = MatrixAdapter.matrixString(input);
		String gradStr = MatrixAdapter.matrixString(g);
		
		System.out.println("INPUT: " + inStr);
		System.out.println("OUT: " + out);
		System.out.println("GRAD: " + gradStr);
	}
	
}
