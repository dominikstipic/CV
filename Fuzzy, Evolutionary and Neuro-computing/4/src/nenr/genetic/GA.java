package nenr.genetic;
import static nenr.main.GAConfigurator.COMPONENTS;
import static nenr.main.GAConfigurator.CROSSOVER;
import static nenr.main.GAConfigurator.GENERATION_NUMBER;
import static nenr.main.GAConfigurator.LOWER;
import static nenr.main.GAConfigurator.MUTATION;
import static nenr.main.GAConfigurator.POP_SIZE;
import static nenr.main.GAConfigurator.SELECTION;
import static nenr.main.GAConfigurator.UPPER;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import nenr.main.Utils;
import nenr.operators.ICrossOver;
import nenr.operators.IMutation;
import nenr.operators.ISelection;

public abstract class GA implements Runnable{
	protected Set<Chromosome> population = new HashSet<>();
	private Function<Integer, Boolean> stopCondition;
	protected final Function<Chromosome, Double> fitness;;
	
	public static Comparator<Chromosome> chromosomeComparator;

	protected IMutation mutation = MUTATION;
	protected ICrossOver crossOver = CROSSOVER;
	protected ISelection selection = SELECTION;
	

	public GA(Function<Chromosome, Double> optim) {
		fitness = optim;
		configureGA();
	}
	
	private void configureGA() {
		int populationSize = POP_SIZE;
		for(int i = 0; i < populationSize; ++i) {
			double bs[] = Utils.randDouble(COMPONENTS, LOWER, UPPER);
			Chromosome chromosome = new Chromosome(bs);
			population.add(chromosome);
		}
		int generations = GENERATION_NUMBER;
		stopCondition = i -> i == generations;
	}
	
	private double populationFitness() {
		Double fitnessValues = population.stream()
				                         .map(c -> fitness.apply(c))
				                         .reduce(0., (a,b) -> a+b);
		return fitnessValues;
	}
	
	
	@Override
	public void run() {
		evolve();
	}

	public List<Double> evolve() {
		int k = 0;
		List<Double> fitness = new ArrayList<>();
		while(!stopCondition.apply(k++)) {
			double populationFitness = populationFitness();
			fitness.add(populationFitness);
			if(k % 2 == 0) {
				int N = GENERATION_NUMBER;
				double p = (double) k / N * 100;
				System.out.format("ITER: %d, PROGRESS: %2f, POPULATION FITNESS: %f\n", k, p, populationFitness);
			}
			Set<Chromosome> newPopulation = geneticStep();
			if(newPopulation.size() != population.size()) 
				throw new IllegalStateException("Calculated population size isn't the same as old population size");
			population = newPopulation;
		}
		return fitness;
	} 
	
	public Chromosome getBest() {
		List<Chromosome> rankedPopulation = population.stream().sorted(chromosomeComparator).collect(Collectors.toList());
		return rankedPopulation.get(0);
	}
	
	protected abstract Set<Chromosome> geneticStep();

}
