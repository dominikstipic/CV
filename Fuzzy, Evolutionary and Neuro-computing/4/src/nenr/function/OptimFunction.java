package nenr.function;

import static java.lang.Math.*;

import java.io.FileNotFoundException;
import java.util.function.BiFunction;
import java.util.function.Function;

import nenr.dataset.Dataset;
import nenr.dataset.IDataset;
import nenr.dataset.Measure;
import nenr.genetic.Chromosome;
import nenr.main.GAConfigurator;
import nenr.main.Utils;

public class OptimFunction implements Function<Chromosome, Double>{
	private IDataset dataset;
	private BiFunction<Double, Double, Double> function;
	public static final double EPSILON = 1e-3;
	
	public OptimFunction(String datasetPath) {
		try {
			this.dataset = Dataset.fromPath(datasetPath);
		} catch (FileNotFoundException e) {
			System.out.println("Dataset "+ datasetPath +" cannot be created");
		}
	}

	@Override
	public Double apply(Chromosome chr) {
		if(chr.componentSize() != GAConfigurator.COMPONENTS) 
			throw new IllegalArgumentException("Optimization function requires chromosomes with 5 components");
		double[] bs = Utils.toPrimitive(chr.getOriginal());
		
		function = (x,y) -> sin(bs[0] + bs[1]*x) + bs[2]*cos(x*(bs[3]*y))*1/(1+exp(pow(x-bs[4],2)));
		double diff=0;
		for(Measure m : dataset) {
			double estimateValue = function.apply(m.getX(), m.getY());
			double realValue = m.getY();
			double residual = estimateValue - realValue;
			diff += pow(residual, 2);
		}
		int N = dataset.size();
		double result = diff/N;
		result = 1/(pow(result+EPSILON, 2));
		return result;
	}
	

}