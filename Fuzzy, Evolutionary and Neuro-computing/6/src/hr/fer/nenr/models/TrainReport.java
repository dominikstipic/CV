package hr.fer.nenr.models;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.github.plot.Plot;

import Jama.Matrix;
import hr.fer.nenr.blocks.ComputationalBlock;
import hr.fer.nenr.dataset.MLDataset;
import hr.fer.nenr.utils.MatrixAdapter;

public class TrainReport {
	public List<Double> train;
	public List<Double> test;
	
	public TrainReport(List<Double> train, List<Double> test) {
		this.train = train;
		this.test = test;
	}
	
	public static TrainReport fromTrain(List<Double> train) {
		return new TrainReport(train, null);
	}
	
	public static TrainReport fromTest(List<Double> test) {
		return new TrainReport(null, test);
	}
	
	
	private static void saveToFile(List<Double> data, Path path) {
		List<String> xs = data.stream().map(d -> String.valueOf(d)).collect(Collectors.toList());
		try {
			Files.write(path, xs, Charset.defaultCharset());
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	public void save(Path root, String name) {
		if(train != null) {
			Path tr = root.resolve(Path.of(name + "_train"));
			saveToFile(train, tr);
		}
		if(test != null) {
			Path tr = root.resolve(Path.of(name + "_test"));
			saveToFile(test, tr);
		}
	}
	
	
	private void plotTest() {
		List<Double> xs = IntStream.range(1, test.size()).boxed().mapToDouble(i -> (double)i).boxed().collect(Collectors.toList());
		Plot plot = Plot.plot(null).series(null, Plot.data().xy(xs, test), null);
			try {
				plot.save("./test_errors", "png");
			} catch (IOException e) {
				e.printStackTrace();
			}
    }
	
	private void plotTrain() {
		List<Double> xs = IntStream.range(1, train.size()).boxed().mapToDouble(i -> (double)i).boxed().collect(Collectors.toList());
		Plot plot = Plot.plot(null).series(null, Plot.data().xy(xs, train), null);
			try {
				plot.save("./train_errors", "png");
			} catch (IOException e) {
				e.printStackTrace();
			}
    }
	
	public void plot() {
		if(test != null) plotTest();
		if(train != null) plotTrain();
    }

	public List<Double> getTrain() {
		return train;
	}


	public List<Double> getTest() {
		return test;
	}

	public static void prediction(ComputationalBlock model, MLDataset dataset, Path root) {
		Path path = root.resolve(Paths.get("predictions"));
		System.out.println(path);
		try(BufferedWriter writer = Files.newBufferedWriter(path)){
			for(Example ex : dataset) {
				List<Double> xs = ex.inputs;
				Matrix input = MatrixAdapter.toVec(xs);
				Matrix pred = model.forward(input);
				List<Double> predValue = MatrixAdapter.fromVec(pred);
				
				double prediction = predValue.get(0);
				double x = xs.get(0); double y = xs.get(1);
				String str = String.format("%s %s %s\n", x,y,prediction);
				writer.write(str);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public static void deltas(ComputationalBlock model, MLDataset dataset, Path root) {
		Path path = root.resolve(Paths.get("deltas"));
		System.out.println(path);
		try(BufferedWriter writer = Files.newBufferedWriter(path)){
			for(Example ex : dataset) {
				List<Double> xs = ex.inputs;
				Matrix input = MatrixAdapter.toVec(xs);
				Matrix pred = model.forward(input);
				List<Double> predValue = MatrixAdapter.fromVec(pred);
				
				double prediction = predValue.get(0);
				double target = ex.label.get(0);
				double x = xs.get(0); double y = xs.get(1);
				double delta = prediction - target;
				String str = String.format("%s %s %s\n", x,y,delta);
				writer.write(str);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	

}
