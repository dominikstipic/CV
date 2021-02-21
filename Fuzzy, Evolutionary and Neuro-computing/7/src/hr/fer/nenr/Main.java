package hr.fer.nenr;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;

import hr.fer.nenr.dataset.Dataset;
import hr.fer.nenr.dataset.IDataset;
import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.EliminationalGA;
import hr.fer.nenr.ga.GA;
import hr.fer.nenr.models.Data;
import hr.fer.nenr.models.NeuralNet;
import hr.fer.nenr.models.Repo;
import hr.fer.nenr.models.TrainReport;
import hr.fer.nenr.utils.Utils;

public class Main {
	
	private static double[] fromFile(GA ga, String path) {
		Path dirName = Paths.get(path);
		List<String> lines = Utils.readLines(dirName);
		double [] best = null; double bestFitness = 0;
		for(String line : lines) {
			double[] list = Utils.stringToList(line);
			double f = ga.calcFitness(new Chromosome(list));
			if(f > bestFitness) {
				bestFitness = f;
				best = list;
			}
		}
		System.out.println("BEST : " + bestFitness);
		return best;
	}
	
	private static String evaluate(IDataset<Data> dataset, NeuralNet net) {
		int N = dataset.size();
		int m = 0;
		StringBuilder sb = new StringBuilder();
		for(Data data : dataset) {
			double[] input = data.example; 
			double[] target = data.oneHot;
			double[] pred = net.predict(input);
			boolean isPredicted = Arrays.compare(pred, target) == 0;
			String str = String.format("PRED = %s, TARGET = %s, %s\n", Arrays.toString(pred), Arrays.toString(target), isPredicted);
			System.out.println(str);
			sb.append(str);
			if(isPredicted) ++m;
		}
		sb.append("---------\n");
		double acc = ((double) m/N)*100.0;
		String str = String.format("ACC = %.3f %%", acc);
		System.out.println(str);
		sb.append(str);
		return sb.toString();
	}
	
	private static List<Double[]> layer1Points(NeuralNet net){
		double[][] G = net.getGroupParam();
		//double[][] S = net.getScaleParam();
		List<Double[]> support = new ArrayList<>();
		for(int i = 0; i < G.length; ++i) {
			double[] g = G[i];
			Double[] g_obj = ArrayUtils.toObject(g);
			support.add(g_obj);
		}
		return support;
	}
	
//	private static List<Double> layer2Weights(NeuralNet net){
//		double[] weights = net.asArray()[1];
//		List<Double> xs = new ArrayList<>();
//		for(int i = 0; i < weights.length; ++i) {
//			Double d = weights[i];
//			xs.add(d);
//		}
//		return xs;
//	}
	
	
	public static void main(String[] args) throws IOException {
		GAConfigurator.build();
		
		IDataset<Data> dataset = Dataset.build();
		NeuralNet net = GAConfigurator.NET;
		GA ga = new EliminationalGA(net, dataset);
		List<Double> errors = ga.evolve();
		
		TrainReport.linePlot(errors);
		Path path = Repo.getLatest();
		net.setParams(fromFile(ga, path.resolve("params").toString()));
		String evalStr = evaluate(dataset, net);
		Files.writeString(path.resolve("eval"), evalStr);
		
		List<Double[]> data = new ArrayList<>();
		List<Integer> label = new ArrayList<>();
		for(Data d : dataset) {
			Double[] arr = org.apache.commons.lang3.ArrayUtils.toObject(d.example);
			data.add(arr); label.add(d.classId());
		}
		
		List<Double[]> learned = layer1Points(net);
		TrainReport.scatter(data, label, learned);
		GAConfigurator.toFile();
		TrainReport.scatter(data, label, learned, new int[] {0,1}, new int[] {0,1});
		
	}

}
