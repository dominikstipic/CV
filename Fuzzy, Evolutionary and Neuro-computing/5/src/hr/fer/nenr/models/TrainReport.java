package hr.fer.nenr.models;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.github.plot.Plot;

public class TrainReport {
	private List<Double> train;
	private List<Double> test;
	
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

	
	

}
