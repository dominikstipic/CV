package nenr.main;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

import com.github.plot.Plot;

import nenr.function.OptimFunction;
import nenr.genetic.Chromosome;
import nenr.genetic.EliminationalGA;
import nenr.genetic.GA;
import nenr.genetic.GenerationalGA;

public class Main {
	public static Function<Chromosome, Double> optim = new OptimFunction("data/dataset1.txt"); 
	
	public static void waitEnd() {
		try(Scanner s = new Scanner(System.in)){
			while(true) {
				String line = s.nextLine().strip();
				if(line.equals("K")) break;
			}
		}
		System.out.println("IZLAZIM");
	}
	
	public static void printBest(GA genAlg) {
		Chromosome best = genAlg.getBest();
		double f = optim.apply(best);
		System.out.println("CHROMOSOME: " + best);
		System.out.println("FITNESS: " + f);
	}
	
	public static void plot(List<Double> ys) {
		List<Double> xss = Utils.arrange(ys.size());
		Plot plot = Plot.plot(null)
				        .series(null, Plot.data().xy(xss, ys), null);
		try {
			plot.save("plot", "png");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		GA genAlg = new EliminationalGA(optim);
		//Thread t = new Thread(genAlg);
		
//		try {
//			t.start();
//			waitEnd();
//			t.stop();
//		} catch (Exception e) {
//			System.out.println("NEŠTO JE LOŠE");
//		}	
		List<Double> fitness = genAlg.evolve();
		plot(fitness);
		printBest(genAlg);
		
	}
}
