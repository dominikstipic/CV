package nenr.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import Jama.Matrix;
import hr.fer.nenr.blocks.ComputationalBlock;
import hr.fer.nenr.blocks.FuzzyLayer;
import hr.fer.nenr.blocks.Linear;
import hr.fer.nenr.blocks.NormalizationLayer;
import hr.fer.nenr.blocks.ParametrizedBlock;
import hr.fer.nenr.blocks.ProductWindow;
import hr.fer.nenr.blocks.Summer;
import hr.fer.nenr.loss.LossFunction;
import hr.fer.nenr.loss.MSE;
import hr.fer.nenr.main.ANFIS;
import hr.fer.nenr.models.Parameters;
import hr.fer.nenr.utils.MatrixAdapter;

public class BlocksTest {

	public void normBlock() {
		NormalizationLayer norm = new NormalizationLayer(2);
		Matrix input = MatrixAdapter.toVec(1.,1.);
		
		Matrix out = norm.forward(input);
		MatrixAdapter.print(out);
		
		Matrix grad = norm.backward(out);
		MatrixAdapter.print(grad);
		
	}

	
	
	
	public void model() {
		Matrix input = MatrixAdapter.toVec(-4,-4);

		FuzzyLayer layer = new FuzzyLayer(2, 4);
		ProductWindow product = new ProductWindow(4,2);
		NormalizationLayer norm = new NormalizationLayer(2);
		Linear linear = new Linear(2, 2);
		Summer sum = new Summer(2);
		
		layer.setParams(Arrays.asList(getParams1()));
		linear.setParams(Arrays.asList(getParams2()));
		
		Matrix out1 = layer.forward(input);
		MatrixAdapter.print(out1); 
		
		Matrix out2 = product.forward(out1);
		MatrixAdapter.print(out2); 
		
		Matrix W = norm.forward(out2);
		MatrixAdapter.print(W);
		
		Matrix F = linear.forward(input);
		MatrixAdapter.print(F);
		
		Matrix h = MatrixAdapter.timesElementwise(F, W);
		Matrix y = sum.forward(h);
		MatrixAdapter.print(y);
		
		
		
		//0.618497362243768 0.3132978461382738  za -4,-4


		
//		Matrix out = MatrixAdapter.replicate(input.get(0, 0), 1, 2);
//		for(int i = 1; i < 2; ++i) {
//			Matrix xs = MatrixAdapter.replicate(input.get(0, i), 1, 2);
//			out = MatrixAdapter.stack(out, xs, true);
//		}
//		
//		Matrix alpha = MatrixAdapter.toVec(1,1,1,1);
//		Matrix g = layer.backward(alpha);
//		MatrixAdapter.print(g);
	}
	
	public void fuzzyBlock() {
		FuzzyLayer layer = new FuzzyLayer(2, 4);
		Matrix input = MatrixAdapter.toVec(1,2);
		Parameters p = getParams1();
		layer.setParams(Arrays.asList(p));
		
		Matrix o = layer.forward(input);
		MatrixAdapter.print(o);
		
		Matrix out = MatrixAdapter.replicate(input.get(0, 0), 1, 2);
		for(int i = 1; i < 2; ++i) {
			Matrix xs = MatrixAdapter.replicate(input.get(0, i), 1, 2);
			out = MatrixAdapter.stack(out, xs, true);
		}
		MatrixAdapter.print(out);
		
		Matrix alpha = MatrixAdapter.toVec(1,1,1,1);
		Matrix g = layer.backward(alpha);
		MatrixAdapter.print(g);
	}
	
	private Parameters getParams1() {
		List<String> names = Arrays.asList("A", "B");
		Matrix a = MatrixAdapter.toVec(0.1, -.3, 0.2, .7);
		Matrix b = MatrixAdapter.toVec(.2, .4, .5, -.1);
		Parameters p = new Parameters(names, Arrays.asList(a,b));
		return p;
	}
	
	private Parameters getParams2() {
		List<String> names = Arrays.asList("W", "b");
		//W = [[p, q],[p, q]]
		Matrix w1 = MatrixAdapter.toVec(.2, -.2);
		Matrix w2 = MatrixAdapter.toVec(.4, -.3);
		Matrix W = MatrixAdapter.stack(w1, w2, false);
		Matrix b = MatrixAdapter.toVec(-.3, -.5);
		Parameters p = new Parameters(names, Arrays.asList(W,b));
		return p;
	}
	
	@Test
	public void anfis() {
		Matrix input = MatrixAdapter.toVec(-4,-4);
		Matrix target = MatrixAdapter.toVec(-23.299);
		
		LossFunction mse = new MSE();

		List<Parameters> params = new ArrayList<>();
		params.add(getParams1());
		params.add(getParams2());
		
		ANFIS anfis = new ANFIS(2);
		anfis.setParams(params);
		
		Matrix output = anfis.forward(input);
		MatrixAdapter.print(output);
		double L = mse.loss(output, target);
		System.out.println("loss: " + L);
		Matrix loss = mse.backward();
		MatrixAdapter.print(loss);
		anfis.backward(loss);
		System.out.println("---------");
		params = anfis.getParams();
		for(Parameters p : params) {
			String gradStr = p.gradString();
			System.out.println(gradStr);
			System.out.println("***");
			//System.out.println(p.gradString());
		}
	}
	
//	public void antBlock() {
//		Matrix input = MatrixAdapter.toVec(-4,-4);
//		int rules = 2;
//		FuzzyLayer layer1 = new FuzzyLayer(2, 2*rules);
//		ProductWindow layer2 = new ProductWindow(2*rules, rules);
//		NormalizationLayer layer3 = new NormalizationLayer(rules);
//		ParametrizedBlock antecedentBlock = new ParametrizedBlock(layer1, layer2, layer3);
//		
//		Matrix y = antecedentBlock.forward(input);
//		MatrixAdapter.print(y);
//		
//		LossFunction mse = new MSE();
//		double loss = mse.loss(MatrixAdapter.fromVec(y), List.of(1.0, 1.0));
//		System.out.println("LOSS: " + loss);
//		
//	}
	
}
