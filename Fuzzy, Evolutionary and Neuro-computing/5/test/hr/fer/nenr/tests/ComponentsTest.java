package hr.fer.nenr.tests;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import Jama.Matrix;
import hr.fer.nenr.blocks.Linear;
import hr.fer.nenr.blocks.Sigmoid;
import hr.fer.nenr.interfaces.ComputationalBlock;
import hr.fer.nenr.interfaces.LossFunction;
import hr.fer.nenr.interfaces.MLDataset;
import hr.fer.nenr.loss.MSE;
import hr.fer.nenr.models.Example;
import hr.fer.nenr.models.Parameters;
import hr.fer.nenr.nn.NN;
import hr.fer.nenr.optim.GradientDescent;
import hr.fer.nenr.optim.Optimizer;

public class ComponentsTest {
	MockDatabase database = new MockDatabase();
	
	
	public void linearTest() {
		Linear linear = new Linear(2, 3);
		double[] w1 = {.5, .2};
		double[] w2 = {.4, .6};
		double[] w3 = {.1, .2};
		double[][] Ws = {w1,w2,w3};
		double[] bias = {0.0, 0.0, 0.0};
		Matrix W = new Matrix(Ws);
		Matrix b = new Matrix(bias, 1);
		
		linear.setParams(new Parameters(W,b));
		
		Parameters p = linear.getParams();
		System.out.println(p);
		System.out.println("-------------");;
		
		Example ex = database.get(0);
		List<Double> output = linear.forward(ex.inputs);
		System.out.println("INPUT: " + ex.inputs);
		System.out.println("LABEL: " + ex.label);
		System.out.println("OUTPUT: " + output);
		
	}
	
	public void linearTest2() {
		double[] w1 = {.3, .35, .45};
		double[] w2 = {.35, .25, .3};
		double[][] Ws = {w1,w2};
		double[] bias = {0.0, 0.0};
		Matrix W = new Matrix(Ws);
		Matrix b = new Matrix(bias, 1);
		
		Linear linear = new Linear(3, 2);
		linear.setParams(new Parameters(W,b));
		
		Parameters p = linear.getParams();
		System.out.println(p);
		System.out.println("-------------");;
		
		List<Double> x = Arrays.asList(.569, .642, .658);
		List<Double> output = linear.forward(x);
		System.out.println("INPUT: " + x);
		System.out.println("OUTPUT: " + output);
		
		List<Double> grad = Arrays.asList(0.07418870953823957, -0.14722370230306192); 
		List<Double> E4 = linear.backward(grad);
		System.out.println("GRADS: " + E4);
		
		List<Double> s3 = Arrays.asList(0.2778873145580292, 0.5829244256019592, 0.6566870808601379);
		ComputationalBlock sigm = new Sigmoid(3);
		sigm.forward(s3);
		
		List<Double> xs = sigm.backward(E4);
		System.out.println(xs);
	}
	
	//TESTED -> CORRECT
	public void sigmTest() {
		ComputationalBlock sigm = new Sigmoid(2);
		List<Double> input = Arrays.asList(0.6916, 0.5571);
		List<Double> output = sigm.forward(input);
		System.out.println("INPUT: " + input);
		System.out.println("OUTPUT: " + output);
		
		List<Double> grad = Arrays.asList(1.0-0.666334, 0.0-0.635793); 
		List<Double> dx = sigm.backward(grad);
		System.out.println("dx: " + dx);
		
	}
	
	//CORRECT
	public void lossAndSigm() {
		ComputationalBlock sigm = new Sigmoid(2);
		List<Double> input = Arrays.asList(0.6916, 0.5571);
		List<Double> output = sigm.forward(input);
		System.out.println("INPUT: " + input);
		System.out.println("OUTPUT: " + output);
		
		Integer i = 0;
		LossFunction mse = new MSE();
		double loss = mse.loss(output, i);
		System.out.println("LOSS: " + loss);
		
		List<Double> lossGrads = mse.backward();
		List<Double> E = sigm.backward(lossGrads);
		System.out.println(E);
//			List<Double> grad = Arrays.asList(1.0-0.666334, 0.0-0.635793); 
//			List<Double> dx = sigm.backward(grad);
//			System.out.println("dx: " + dx);
	}
	
	private NN createNN() {
		double[] w1 = {.5, .2};
		double[] w2 = {.4, .6};
		double[] w31 = {.1, .2};
		double[] bias1 = {0.0, 0.0, 0.0};
		
		double[] w3 = {.1, .2, .25};
		double[] w4 = {.55, .45, .15};
		double[] w5 = {.35, .35, .6};
		double[] bias2 = {0.0, 0.0, 0.0};
		
		double[] w6 = {.3, .35, .45};
		double[] w7 = {.35, .25, .3};
		double[] bias3 = {0.0, 0.0};
		
		Matrix W1 = new Matrix(new double[][]{w1, w2, w31});
		Matrix b1 = new Matrix(bias1, 1);
		Matrix W2 = new Matrix(new double[][]{w3, w4, w5});
		Matrix b2 = new Matrix(bias2, 1);
		Matrix W3 = new Matrix(new double[][]{w6, w7});
		Matrix b3 = new Matrix(bias3, 1);
		
		Linear linear1 = new Linear(2, 3);
		linear1.setParams(new Parameters(W1,b1));
		Linear linear2 = new Linear(3, 3);
		linear2.setParams(new Parameters(W2,b2));
		Linear linear3 = new Linear(3, 2);
		linear3.setParams(new Parameters(W3,b3));
		 
		NN nn = new NN(linear1, linear2, linear3);
		
		return nn;
	}
	
	public void nnTest() {
		NN nn = createNN();
		
		List<Double> x = Arrays.asList(.05, .02);
		List<Integer> y = Arrays.asList(1, 0);
		List<Double> output = nn.forward(x);
		System.out.println("INPUT: " + x);
		System.out.println("OUTPUT: " + output);
		
		LossFunction mse = new MSE();
		double loss = mse.loss(output, y);
		System.out.println("LOSS: " + loss);
		
		List<Double> lossGrads = mse.backward();
		List<Double> g = nn.backward(lossGrads);
		System.out.println(g);
		
		
	}
	
	@Test
	public void nnTest1() {
		NN nn = createNN();
		LossFunction mse = new MSE();
		Optimizer optim = new GradientDescent(nn, .3);
		
		List<Double> input = Arrays.asList(.05, .02); List<Integer> label = Arrays.asList(1, 0);
		List<Double> prediction = nn.forward(input);
		double lossVal = mse.loss(prediction, label);
		System.out.println("LOSS: " + lossVal);
		List<Double> dLoss_dPred = mse.backward();
		nn.backward(dLoss_dPred);
		optim.step();
		System.out.println("--------");
		
		nn.getParams();
	}
	
	
	
}
