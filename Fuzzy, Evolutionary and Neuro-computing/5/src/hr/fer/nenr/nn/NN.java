package hr.fer.nenr.nn;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.blocks.Component;
import hr.fer.nenr.blocks.Linear;
import hr.fer.nenr.blocks.Sigmoid;
import hr.fer.nenr.interfaces.ComputationalBlock;
import hr.fer.nenr.interfaces.LossFunction;
import hr.fer.nenr.interfaces.MLDataset;
import hr.fer.nenr.loss.MSE;
import hr.fer.nenr.models.Example;
import hr.fer.nenr.models.Parameters;
import hr.fer.nenr.models.TrainReport;
import hr.fer.nenr.optim.GradientDescent;
import hr.fer.nenr.optim.Optimizer;
import hr.fer.nenr.utils.Label;
import hr.fer.nenr.utils.MatrixAdapter;

public class NN extends Component{
	
	public NN(Integer ...dims) {
		super(buildComponents(dims));
	}
	
	public NN(Linear...layers) {
		super();
		for(Linear layer : layers) {
			children.add(layer);
			children.add(new Sigmoid(layer.outFeatures()));
		}
		in = layers[0].inputFeatures();
		out = layers[layers.length-1].outFeatures();
	}
	
	public NN(List<Integer> dims) {
		this(dims.toArray(new Integer[1]));
	}

	private static List<ComputationalBlock> buildComponents(Integer[] dims){
		List<ComputationalBlock> children = new ArrayList<>();
		for(int i = 0; i < dims.length-1; ++i) {
			ComputationalBlock s = new Linear(dims[i], dims[i+1]);
			children.add(s);
			children.add(new Sigmoid(dims[i+1]));
		}
		return children;
	}
	
	public static NN build(Linear ...layers){
		return new NN(layers);
	}
	
	public List<Integer> dimensions(){
		List<Integer> dims = new ArrayList<>();
		dims.add(children.get(0).inputFeatures());
		for(ComputationalBlock block : children) {
			if(!block.hasParam()) continue;
			int out = block.outFeatures();
			dims.add(out);
		}
		return dims;
	}

	@Override
	public void cleanGradients() {
		children.forEach(c -> {
			if(c.hasParam()) c.cleanGradients();
		});
	}

	@Override
	public boolean hasParam() {
		return true;
	}

	@Override
	public Parameters getParams() {
		for(ComputationalBlock block : children) {
			if(block.hasParam()) {
				Parameters params = block.getParams();
				System.out.println(MatrixAdapter.matrixString(params.W));
			}
		}
		return null;
	}

	@Override
	public Matrix[] getGrads() {
		return null;
	}

	@Override
	public void setParams(Parameters params) {}
	
	
	public Label argmax(List<Double> xs) {
		double max = 0;
		int idx = 0;
		for(int i = 0;  i < xs.size(); ++i) {
			double d = xs.get(i);
			if(d > max) {
				max = d;
				idx = i;
			}
		}
		Label label = Label.fromValue(idx);
		return label;
	}
	
	public TrainReport train(int epochs, MLDataset dataset, Optimizer optim, LossFunction loss, int sampleSize) {
		TrainReport report = train(epochs, dataset, null, optim, loss, sampleSize);
		return report;
	}
	
	public TrainReport train(int epochs, MLDataset trainDb, MLDataset validDb, Optimizer optim, LossFunction loss, int sampleSize) {
		List<Double> trainErrors = new ArrayList<>();
		List<Double> validErrors = new ArrayList<>();
		for(int epoch = 0; epoch < epochs; ++epoch) {
			System.out.println("EPOCH " + epoch);
			cleanGradients();
			MLDataset sample = trainDb.sample(sampleSize);
			double train_err = 0;
			for(int i = 0; i < sampleSize; ++i) {
				Example example = sample.get(i);
				List<Double> input = example.inputs; Integer label = example.label;
				List<Double> prediction = forward(input);
				train_err += loss.loss(prediction, label);
				List<Double> dLoss_dPred = loss.backward();
				backward(dLoss_dPred);
			}
			train_err /= sampleSize;	
			double valid_err = 0;
			if(validDb != null) valid_err = validate(validDb, loss);
			trainErrors.add(train_err);
			validErrors.add(valid_err);
			
			String msg;
			if(validDb != null)
				msg = String.format("TRAIN: %s, TEST: %s", train_err, valid_err);
			else 
				msg = String.format("TRAIN: %s", train_err);
			System.out.println(msg);
			System.out.println("--------");
			optim.step();
		}
		if(validDb != null)
			return new TrainReport(trainErrors, validErrors);
		else 
			return TrainReport.fromTrain(trainErrors);
	}
	
	public double validate(MLDataset dataset, LossFunction loss) {
		double error = 0;
		for(Example example : dataset) {
			List<Double> input = example.inputs; Integer label = example.label;
			List<Double> prediction = forward(input);
			error += loss.loss(prediction, label);
		}
		error /= dataset.size();
		return error;
	}
	
	public static void main(String[] args) {
		int M = 20;
		double epsilon = 0.01;
		LossFunction mse = new MSE();
		NN net = new NN(2*M, 3, 2, 5);
		Optimizer optim = new GradientDescent(net, epsilon);
		
		NeuralDataset dataset = new NeuralDataset(NeuralDataset.PATH, M);
		MLDataset[] arr = dataset.split(0.7);
		MLDataset trainDataset = arr[0]; MLDataset testDataset = arr[1];
		System.out.println(trainDataset.size() + "," + testDataset.size());
		
//		TrainReport report = net.train(100, dataset, optim, mse, 100);
		TrainReport report = net.train(100, trainDataset, testDataset, optim, mse, 70);
		report.plot();	
		
		

	}
	
	
}
