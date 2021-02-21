package hr.fer.nenr.main;

import java.awt.Color;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

import com.github.plot.Plot;
import com.github.plot.Plot.DataSeriesOptions;
import com.github.plot.Plot.Line;
import com.github.plot.Plot.PlotOptions;

import Jama.Matrix;
import hr.fer.nenr.blocks.ComputationalBlock;
import hr.fer.nenr.dataset.Dataset;
import hr.fer.nenr.dataset.Functions;
import hr.fer.nenr.dataset.MLDataset;
import hr.fer.nenr.loss.LossFunction;
import hr.fer.nenr.loss.MSE;
import hr.fer.nenr.models.Example;
import hr.fer.nenr.models.Parameters;
import hr.fer.nenr.models.TrainReport;
import hr.fer.nenr.optim.GradientDescent;
import hr.fer.nenr.optim.Optimizer;
import hr.fer.nenr.train.Trainer;
import hr.fer.nenr.utils.MatrixAdapter;
import static java.awt.Color.*;

public class Main {
	private static Path ROOT = Paths.get("./files");
	private static Color[] colors = {RED, BLACK, BLUE, GREEN, ORANGE};
	
	public static Runnable job = () -> {
		try(Scanner s = new Scanner(System.in)){
			while(true) {
				String line = s.nextLine().trim();
				line = line.toUpperCase();
				if(line.equals("END")) break;
			}
		}
		Trainer.active = false;
	};
	
	
	
	
	public static void membershipTest() {
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		double epsilon = 1E-5;
		int epochs = (int) 1E5;
		LossFunction mse = new MSE();
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();
		
		for(int i = 1; i <= 3; ++i) {
			System.out.println("rule: " + i);
			ANFIS anfis = new ANFIS(i);
			Optimizer optim = new GradientDescent(anfis, epsilon);
			Trainer.train(anfis, epochs, dataset, optim, mse, dataset.size());
			String path = "./plots/"+ "R"+i;
			String title = "RULE " + i;
			plot(title, path, anfis);
		}
	}
	
	private static void plot(String title, String path, ComputationalBlock anfis) {
		List<Parameters> params = anfis.getParams();
		Matrix A = params.get(0).getParam("A");
		Matrix B = params.get(0).getParam("B");
		List<Double> as = MatrixAdapter.fromVec(A);
		List<Double> bs = MatrixAdapter.fromVec(B);
		
		System.out.println("PARAM SIZE: " + as.size());
		PlotOptions options = Plot.plotOpts().title(title).legend(Plot.LegendFormat.BOTTOM);
		DataSeriesOptions seriesOptions = Plot.seriesOpts().color(red).line(Line.SOLID);
		for(int i = 0; i < as.size(); ++i) {
			double a = as.get(i); double b = bs.get(i);
			List<List<Double>> data = new Sampler(a, b).sample(-4, 4, 50);
			List<Double> xs = new ArrayList<>(); List<Double> ys = new ArrayList<>();
			data.forEach(list ->   {
				double x = list.get(0); 
				double y = list.get(1);
				xs.add(x); ys.add(y);
			});
			Plot plot = Plot.plot(options).
					         xAxis("x", Plot.axisOpts().range(-4, 4)).
					         yAxis("mi(x)", Plot.axisOpts().range(0, 1)).
					         series("data", Plot.data().xy(xs, ys), seriesOptions);
			try {plot.save(path + "_N"+i  , "png");} 
			catch (IOException e) {e.printStackTrace();}
		}
	}
	
	public static void test1() {
		int rules = 2;
		ANFIS anfis = new ANFIS(rules);
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		
		double epsilon = 1E-4;
		int epochs = (int) 1E5;
		int batch = dataset.size();
		LossFunction mse = new MSE();
		Optimizer optim = new GradientDescent(anfis, epsilon);
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();
		
		TrainReport report = Trainer.train(anfis, epochs, dataset, optim, mse, batch);
		//report.plot();
		//report.save(ROOT);
	}
	
	public static void test2() {
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		MLDataset[] arr = dataset.split(0.7);
		MLDataset trainDb = arr[0];
		MLDataset testDb = arr[1];
		
		List<Double> epsilons = Arrays.asList(5E-3, 3E-3, 1E-3, 
				                              5E-4, 3E-4, 1E-4, 
				                              5E-5, 3E-5, 1E-5,
				                              5E-6, 3E-6, 1E-6);
		int epochs = (int) 1E4;
		int batch = 1;
		LossFunction mse = new MSE();
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();
		
		for(Double epsilon : epsilons) {
			int rules = 4;
			ANFIS anfis = new ANFIS(rules);
			Optimizer optim = new GradientDescent(anfis, epsilon);
			TrainReport report = Trainer.train(anfis, epochs, trainDb, testDb, optim, mse, batch);
			String name = "lr/Epsilon_" + epsilon;
			System.out.println(epsilon + " " + report.test.get(epochs-1));
			//report.save(ROOT, name);
		}
		
		
		
		
		//report.plot();
		
	}
	
	public static void test3() {
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		MLDataset[] arr = dataset.split(0.7);
		MLDataset trainDb = arr[0];
		MLDataset testDb = arr[1];
		double epsilon = 1E-5;
		int epochs = (int) 1E4;
		int batch = trainDb.size();
		LossFunction mse = new MSE();
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();

		for(int i = 1; i <= 5; ++i) {
			ANFIS anfis = new ANFIS(i);
			Optimizer optim = new GradientDescent(anfis, epsilon);
			TrainReport report = Trainer.train(anfis, epochs, trainDb, testDb, optim, mse, batch);
			int trainSize = report.train.size();
			int testSize = report.test.size();
			String str = String.format("%s %s %s", i, report.train.get(trainSize-1), report.test.get(testSize-1));
			System.out.println(str);
		}
		
	}
	
	public static void test4() {
		int rules = 2;
		ANFIS anfis = new ANFIS(rules);
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		double epsilon = 1E-4;
		int epochs = (int) 1E4;
		LossFunction mse = new MSE();
		Optimizer optim = new GradientDescent(anfis, epsilon);
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();

		Trainer.train(anfis, epochs, dataset, optim, mse, dataset.size());
		TrainReport.prediction(anfis, dataset, ROOT);	
	}
	
	public static void test5() {
		int rules = 2;
		ANFIS anfis = new ANFIS(rules);
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		double epsilon = 1E-4;
		int epochs = (int) 1E3;
		LossFunction mse = new MSE();
		Optimizer optim = new GradientDescent(anfis, epsilon);
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();

		Trainer.train(anfis, epochs, dataset, optim, mse, dataset.size());
		TrainReport.deltas(anfis, dataset, ROOT);	
	}
	
	public static void test7() {
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		List<Integer> batches = Arrays.asList(1, 10, 20, 30, 40, 50, 60, 70, 80, dataset.size());
		int epochs = (int) 1E4;
		double epsilon = 1E-5;
		LossFunction mse = new MSE();
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();
		
		for(int batch : batches) {
			int rules = 3;
			ANFIS anfis = new ANFIS(rules);
			Optimizer optim = new GradientDescent(anfis, epsilon);
			TrainReport report = Trainer.train(anfis, epochs, dataset, optim, mse, batch);
			System.out.println(batch + " " + report.train.get(epochs-1));
			//report.save(ROOT, name);
		}
		
		
		
		
		//report.plot();
		
	}
	
	public static void test8() {
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		MLDataset[] arr = dataset.split(0.7);
		MLDataset trainDb = arr[0];
		MLDataset testDb = arr[1];
		
		List<Integer> batches = Arrays.asList(10, 20, trainDb.size());
		int epochs = (int) 1E4;
		double epsilon = 1E-5;
		LossFunction mse = new MSE();
		
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();
		
		for(int batch : batches) {
			int rules = 2;
			ANFIS anfis = new ANFIS(rules);
			Optimizer optim = new GradientDescent(anfis, epsilon);
			TrainReport report = Trainer.train(anfis, epochs, trainDb, testDb, optim, mse, batch);
			System.out.println(batch + " " + report.train.get(epochs-1));
			String name = "B"+batch;
			report.save(ROOT, name);
		}
	}
	
	public static void evaluateRules() {
		int rules = 4;
		ANFIS anfis = new ANFIS(rules);
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		double epsilon = 1E-4;
		int epochs = (int) 3E3;
		LossFunction mse = new MSE();
		Optimizer optim = new GradientDescent(anfis, epsilon);
		
		Thread t = new Thread(job);
		t.setDaemon(true);
		t.start();

		Trainer.train(anfis, epochs, dataset, optim, mse, dataset.size());
		
		Path path = ROOT.resolve(Paths.get("activations"));
		try(BufferedWriter writer = Files.newBufferedWriter(path)){
			int k = 0;
			for(Example ex : dataset) {
				if(k % 9 == 0) {
					writer.write("\n");
				}
				++k;
				List<Double> xs = ex.inputs;
				Matrix input = MatrixAdapter.toVec(xs);
				List<Double> activations = anfis.evaluateRules(input);

				double x = xs.get(0);
				double y = xs.get(1);
				String str = String.format("%s %s ", x,y);
				String act = activations.stream().map(d -> String.valueOf(d)).collect(Collectors.joining(" ", "", ""));
				str = str + act;
				writer.write(str+"\n");
				
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	public static void main(String[] args) {
		test2();
	}
	
	///////////////////
	
	public static class Sampler{
		TriFuncion<Double, Double, Double, Double> mi = (x, a, b) -> 1/(1+Math.exp(b*(x-a)));
		public double a; public double b;
		
		public Sampler(double a, double b) {
			this.a = a; this.b = b;
		}
		
		public List<List<Double>> sample(int from, int to, int n){
			double d = Math.abs(from) + Math.abs(to);
			d = d/(n-1);
			List<List<Double>> xs = new ArrayList<>();
			for (int i = 0; i < n; ++i){
				double x = from + i*d;
				double y = mi.apply(x, a, b);
				xs.add(Arrays.asList(x,y));
			}
			return xs;
		}
	}
	
	public interface TriFuncion<R,S,T,V>{
		V apply(R r, S s, T t);
	}

}
