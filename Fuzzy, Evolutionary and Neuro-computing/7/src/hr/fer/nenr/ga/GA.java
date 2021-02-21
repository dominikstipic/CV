package hr.fer.nenr.ga;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import hr.fer.nenr.GAConfigurator;
import hr.fer.nenr.dataset.IDataset;
import hr.fer.nenr.ga.operators.ICrossOver;
import hr.fer.nenr.ga.operators.IMutation;
import hr.fer.nenr.ga.operators.ISelection;
import hr.fer.nenr.models.Data;
import hr.fer.nenr.models.NeuralNet;
import hr.fer.nenr.models.Repo;
import hr.fer.nenr.utils.Utils;
import static hr.fer.nenr.GAConfigurator.*;

public abstract class GA{
	public List<Chromosome> population = new LinkedList<>();
	protected double populationFitness;
	protected Function<Double, Double> fitness;
	private Function<Integer, Boolean> stopCondition;
	private NeuralNet net;
	private IDataset<Data> dataset;
	public static final int PRINT_PERIOD = POP_SIZE;
	private Path saveFile;

	protected IMutation mutation = MUTATION;
	protected ICrossOver crossOver = CROSSOVER;
	protected ISelection selection = SELECTION;
	
	public GA(NeuralNet net, IDataset<Data> dataset) {
		this.net = net;
		this.dataset = dataset;
		configureGA();
		createDir();
	}
	
	private void createDir() {
		Path dir = Repo.getNext();
		saveFile = dir.resolve("params");
		try {
			Files.createDirectories(dir);
			Files.createFile(saveFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	private void configureGA() {
		int populationSize = POP_SIZE;
		fitness = GAConfigurator.fitnessFunction;
		for(int i = 0; i < populationSize; ++i) {
			double bs[] = Utils.randomDoubleVector(COMPONENTS, LOWER, UPPER);
			Chromosome chromosome = new Chromosome(bs);
			population.add(chromosome);
		}
		Comparator<Chromosome> chromosomeComparator = (c1, c2) -> { 
			boolean v1 = calcFitness(c1) > calcFitness(c2);
			boolean v2 = calcFitness(c1) < calcFitness(c2);
			if(v1) return -1;
			else if(v2) return 1;
			else return 0;};
		population = population.stream().sorted(chromosomeComparator).collect(Collectors.toList());
		populationFitness = population.stream()
                                      .map(c -> calcFitness(c))
                                      .reduce(0., (a,b) -> a+b);
		stopCondition = i -> i == GENERATION_NUMBER*POP_SIZE;
	}
	
	protected void rankedInsert(Chromosome chromosome) {
		int index = 0;
		double chrFitness = calcFitness(chromosome);
		for(index = 0; index < population.size(); ++index) {
			double currentFitness = calcFitness(population.get(index));
			if(chrFitness >= currentFitness) {
				break;
			}
		}
		population.add(index, chromosome);
	}
	
	public double calcFitness(Chromosome chromosome) {
		double mse = net.calcError(dataset, chromosome.solution);
		double fitnessValue = fitness.apply(mse);
		return fitnessValue;
	}
	
	public Chromosome getBest() {
		return population.get(0);
	}
	
	private void periodicJob(int iterNum) {
		int N = GENERATION_NUMBER;
		int generation = iterNum / POP_SIZE;
		double p = ((double) generation / N);
		double best = calcFitness(getBest());
		System.out.format("PERCENTAGE: %.3f, " + 
				          "GENERATION: %d ," + 
				          "POPULATION FITNESS: %.3f, " + 
				          "BEST FITNESS: %.3f", 
				           p*100, generation, populationFitness, best);
		System.out.println();
		
		double[] bestGenom = getBest().solution;
		String line = Arrays.toString(bestGenom) + "\n";
		Utils.appendToFile(line, saveFile);
	}
	
	public void printPopulation() {
		String s = "";
		for(Chromosome c : population) {
			Double f = calcFitness(c);
			s += f + ", ";
		}
		System.out.println(s);
	}

	public List<Double> evolve() {
		int k = 0;
		List<Double> fitnessProcess = new ArrayList<>();
		while(!stopCondition.apply(k++)) {
			fitnessProcess.add(populationFitness);
			if(k % PRINT_PERIOD == 0) periodicJob(k);
			geneticStep();
		}
		return fitnessProcess;
	} 
	
	protected abstract void geneticStep();

}
