package nenr.main;

import java.util.function.Function;

import nenr.function.OptimFunction;
import nenr.genetic.Chromosome;
import nenr.mutation.GaussianMutation;
import nenr.mutation.UniformMutation;
import nenr.operators.ICrossOver;
import nenr.operators.IMutation;
import nenr.operators.ISelection;
import nenr.reproduction.FloatReproduction;
import nenr.selection.Torurnament;

public class GAConfigurator {
	public static double LOWER = -4;
	public static double UPPER = 4;
	public static int COMPONENTS = 5;
	public static int GENERATION_NUMBER = 5000; 
	public static int POP_SIZE = 200;
	public static boolean ELITISM = true;
	public static Function<Chromosome, Double> OPTIM_FUN = new OptimFunction("data/dataset2.txt"); 
	
	//OPERATORS
	public static IMutation MUTATION = new UniformMutation();
	public static ICrossOver CROSSOVER = new FloatReproduction();
	public static ISelection SELECTION = new Torurnament(3);
	
	public static double MUTATION_PROBABILITY = 0.1;
	public static double CROSSOVER_PROBABILITY = 0.8;
}
