package nenr.operators;

import java.util.Comparator;
import java.util.List;
import java.util.Set;
import nenr.genetic.Chromosome;

public interface ISelection {
	List<Chromosome> select(int n, Set<Chromosome> population, Comparator<Chromosome> chromosomeComparator);
}
