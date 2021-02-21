package hr.fer.nenr;

import static java.lang.String.format;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import hr.fer.nenr.ga.crossOver.WeightCrossOver;
import hr.fer.nenr.ga.mutations.AdditiveGaussianNoise;
import hr.fer.nenr.ga.mutations.GaussianNoise;
import hr.fer.nenr.ga.mutations.MutationComposition;
import hr.fer.nenr.ga.operators.ICrossOver;
import hr.fer.nenr.ga.operators.IMutation;
import hr.fer.nenr.ga.operators.ISelection;
import hr.fer.nenr.ga.selection.TournamentSelection;
import hr.fer.nenr.models.NeuralNet;
import hr.fer.nenr.models.Repo;
import hr.fer.nenr.utils.Utils;

public class GAConfigurator {
	public static Function<Double, Double> fitnessFunction = x -> 1/x;
	public static double LOWER = -1;
	public static double UPPER = 1;
	public static int GENERATION_NUMBER = 1000; 
	public static int POP_SIZE = 20;
	public static int COMPONENTS;
	public static int TOURNAMENT_SIZE = 3;
	public static double[] TS = {5,3,1};
	public static double[] SIGMAS = {0.2, .4, 1};
	public static double[] PROBS  = {0.1, 0.05, 0.001};
	public static int[] DIMS = {2,8,3};
	
	
	//OPERATORS
	public static IMutation MUTATION;
	public static ICrossOver CROSSOVER;
	public static ISelection SELECTION;
	
	public static NeuralNet NET;
	
	{
		NET = new NeuralNet(DIMS);
		COMPONENTS = NET.paramSize;
		
		//CROSSOVER = new DummyCrossOver();
		CROSSOVER = new WeightCrossOver(0.1);
		//CROSSOVER = new NeuronCrossOver(NET);
		//CROSSOVER = new FeatureCrossOver(NET);
		
		SELECTION = new TournamentSelection(TOURNAMENT_SIZE);
		
		//MUTATION = new DummyMutation();
		//MUTATION = new AdditiveGaussianNoise(2, 0.001);
		MUTATION = new MutationComposition(ls(new AdditiveGaussianNoise(SIGMAS[0], PROBS[0]),
                                              new AdditiveGaussianNoise(SIGMAS[1], PROBS[1]),
                                              new GaussianNoise(SIGMAS[2], PROBS[2])), TS);
	}
	
	public static String info() {
		StringBuffer sb = new StringBuffer();
		sb.append(format("DIM=%s\n", Arrays.toString(DIMS)));
		sb.append(format("LOWER=%s, UPPER=%s\n", LOWER, UPPER));
		sb.append(format("GENERATIONS=%s\n", GENERATION_NUMBER));
		sb.append(format("POP_SIZE=%s\n", POP_SIZE));
		sb.append(format("COMPONENTS=%s\n", COMPONENTS));
		sb.append(format("TOURNAMENT_SIZE=%s\n", TOURNAMENT_SIZE));
		sb.append(format("SIGMAS=%s\n", Arrays.toString(SIGMAS)));
		sb.append(format("PROBS=%s\n", Arrays.toString(PROBS)));
		sb.append(format("CROSS_OVER=%s\n", CROSSOVER.name()));
		return sb.toString();
	}
	
	public static void toFile() {
		Path path = Repo.getLatest().resolve("config");
		try {
			Files.createFile(path);
		} catch (IOException e) {
			e.printStackTrace();
		}
		String config = info();
		Utils.appendToFile(config, path);
	}
	
	private GAConfigurator() {}
	
	public static void build() {
		new GAConfigurator();
	}

	
	private static List<IMutation> ls(IMutation ...muts) {
		List<IMutation> list = new ArrayList<>();
		for(IMutation m : muts) list.add(m);
		return list;
	}
}
