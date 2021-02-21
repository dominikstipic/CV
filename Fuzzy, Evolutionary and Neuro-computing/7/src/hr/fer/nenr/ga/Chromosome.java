package hr.fer.nenr.ga;
import java.util.Arrays;
import java.util.Objects;

import hr.fer.nenr.utils.Utils;

public class Chromosome {
	public final double[] solution;
	
	public Chromosome(double ...xs) {
		solution = xs;
	}
	
	public Chromosome(int n) {
		solution = new double[n];
		for(int i = 0; i < n; ++i) {
			solution[i] = Utils.randomDouble(0, 1);
		}
	}
	
	public Chromosome copy() {
		double[] sol = Arrays.copyOf(solution, solution.length);
		return new Chromosome(sol);
	}
	
	@Override
	public int hashCode() {
		return Objects.hash(solution);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Chromosome other = (Chromosome) obj;
		return Arrays.equals(solution, other.solution);
	}

	@Override
	public String toString() {
		return Arrays.toString(solution);
	}
	
}