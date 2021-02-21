package nenr.genetic;
import java.util.ArrayList;
import java.util.List;

public class Chromosome {
//	private double lower = LOWER;
//	private double upper = UPPER;
	private List<Double> original = new ArrayList<>();
	//private BiFunction<Double, Double, Double> fitness;
	
	public Chromosome(double ...xs) {
		for(double x : xs) {
			original.add(x);
		}
	}
	
	public List<Double> getOriginal() {
		return original;
	}
	
	public void setOriginal(List<Double> values) {
		original = values;
	}
	
	public int componentSize() {
		return original.size();
	}
	
	public double get(int idx) {
		if(idx < 0 || idx >= componentSize()) throw new IndexOutOfBoundsException("chromosome components doesn't exist");
		return original.get(idx);
	}

	@Override
	public int hashCode() {
		return original.hashCode();
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
		return other.original.equals(original);
	}

	@Override
	public String toString() {
		return original.toString();
	}
	
}
