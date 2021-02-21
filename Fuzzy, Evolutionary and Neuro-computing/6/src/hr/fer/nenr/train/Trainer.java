package hr.fer.nenr.train;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.blocks.Composite;
import hr.fer.nenr.blocks.ComputationalBlock;
import hr.fer.nenr.dataset.MLDataset;
import hr.fer.nenr.loss.LossFunction;
import hr.fer.nenr.models.Example;
import hr.fer.nenr.models.TrainReport;
import hr.fer.nenr.optim.Optimizer;
import hr.fer.nenr.utils.MatrixAdapter;

public class Trainer {
	public static boolean active = true;
	public static Double EPSILON = 1E-6;
	public static boolean trace = false;
	
	

	public static TrainReport train(ComputationalBlock model, int epochs, MLDataset trainDb, MLDataset validDb, Optimizer optim, LossFunction loss, int sampleSize) {
		List<Double> trainErrors = new ArrayList<>();
		List<Double> validErrors = new ArrayList<>();
		for(int epoch = 0; epoch < epochs; ++epoch) {
			if(trace) System.out.println("EPOCH " + epoch);
			model.cleanGradients();
			MLDataset sample = trainDb.sample(sampleSize);
			double train_err = 0;
			for(int i = 0; i < sampleSize; ++i) {
				Example example = sample.get(i);
				Matrix input = example.getInput(); List<Double> label = example.label;
				Matrix prediction = model.forward(input);
				train_err += loss.loss(prediction, MatrixAdapter.toVec(label));
				Matrix dLoss_dPred = loss.backward();
				model.backward(dLoss_dPred);
			}
			train_err /= sampleSize;	
			double valid_err = 0;
			if(validDb != null) valid_err = validate(model, validDb, loss);
			trainErrors.add(train_err);
			validErrors.add(valid_err);
			if(!active) break;
			String msg;
			if(validDb != null)
				msg = String.format("TRAIN: %s, TEST: %s", train_err, valid_err);
			else 
				msg = String.format("<TRAIN: %s", train_err);
			if(trace) {
				System.out.println(msg);
				System.out.println("--------");
			}
			optim.step();
			
//			double currentError = trainErrors.get(epoch);
//			if(Math.abs(currentError - oldError) <= EPSILON) break;
//			oldError = currentError;
		}
		if(validDb != null)
			return new TrainReport(trainErrors, validErrors);
		else 
			return TrainReport.fromTrain(trainErrors);
	}
	
	public static TrainReport train(Composite model, int epochs, MLDataset dataset, Optimizer optim, LossFunction loss, int sampleSize) {
		TrainReport report = train(model, epochs, dataset, null, optim, loss, sampleSize);
		return report;
	}
	
	
	public static double validate(ComputationalBlock model, MLDataset dataset, LossFunction loss) {
		double error = 0;
		for(Example example : dataset) {
			Matrix input = example.getInput(); List<Double> label = example.label;
			Matrix prediction = model.forward(input);
			error += loss.loss(prediction, MatrixAdapter.toVec(label));
		}
		error /= dataset.size();
		return error;
	}
	
}
