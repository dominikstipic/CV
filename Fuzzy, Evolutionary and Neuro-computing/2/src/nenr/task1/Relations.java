package nenr.task1;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import nenr.zad1.CompositeDomain;
import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;
import nenr.zad2.IFuzzySet;
import nenr.zad2.MutableFuzzySet;

public class Relations {
	public static double EPSILON = 10e-6;

	private static DomainElement standard_index(IDomain domain, int i, int j) {
		int n = domain.getComponent(0).getCardinality();
		int m = domain.getComponent(1).getCardinality();
		int index;
		if(n > m) {
			int delta = n - m;
			index = i*n + j - delta*i;
			
		}
		else if(n < m) index = i*m + j;
		else index = i*n + j;
		return domain.elementForIndex(index);
	}
	
	private static double maxMin(IFuzzySet a, IFuzzySet b, int i, int j) {
		IDomain X = a.getDomain();
		IDomain Y = b.getDomain();
		
		int a_width  = a.getDomain().getComponent(1).getCardinality();
		int b_height = b.getDomain().getComponent(0).getCardinality();
		if(a_width != b_height) throw new IllegalArgumentException("Something got wrong!");

		List<Double> mins = new ArrayList<>();
		for(int k = 0; k < a_width; ++k) {
			DomainElement d1 = standard_index(X, i, k);
			DomainElement d2 = standard_index(Y, k, j);
			double valueA = a.getValueAt(d1);
			double valueB = b.getValueAt(d2);
			double min = Math.min(valueA, valueB);
			mins.add(min);
		}
		return mins.stream().max((d1,d2) -> Double.compare(d1, d2)).get().doubleValue();
	}
	
	private static IFuzzySet calcMaxMinProduct(IFuzzySet a, IFuzzySet b) {
		IDomain X = a.getDomain().getComponent(0);
		IDomain Z = b.getDomain().getComponent(1);
		int a_height = a.getDomain().getComponent(0).getCardinality();
		int b_width  = b.getDomain().getComponent(1).getCardinality();
		
		IDomain result_domain = CompositeDomain.combine(X, Z);
		MutableFuzzySet result = new MutableFuzzySet(result_domain);
		
		for(int i = 0; i < a_height; ++i) {
			for(int j = 0; j < b_width; ++j) {
				double maxmin = maxMin(a, b, i, j);
				DomainElement de = standard_index(result_domain, i, j);
				result.set(de, maxmin);
			}
		}
		return result;
	}
	
	public static boolean isUtimesURelation(IFuzzySet set) {
		IDomain domain = set.getDomain();
		return domain.getNumberOfComponents() == 2;
	}
	
	public static boolean isSymmetric(IFuzzySet relation) {
		if (!isUtimesURelation(relation)) {
			throw new IllegalArgumentException("Argument isn't binary relation");
		}
		IDomain domain = relation.getDomain();
		for(DomainElement elemA : domain) {
			DomainElement elemB = DomainElement.of(elemA.getComponentValue(1),
					                               elemA.getComponentValue(0));
			Double valueA = relation.getValueAt(elemA);
			Double valueB = relation.getValueAt(elemB);
			if(!valueA.equals(valueB)) return false;
			
		}
		return true;
	}
	
	public static boolean isReflexive(IFuzzySet relation) {
		if (!isUtimesURelation(relation)) {
			throw new IllegalArgumentException("Argument isn't binary relation");
		}
		IDomain domain = relation.getDomain();
		int n = (int) Math.sqrt(domain.getCardinality());
		for(int i = 0; i < n; ++i) {
			int idx = n * i + i;
			DomainElement element = domain.elementForIndex(idx);
			Double value = relation.getValueAt(element);
			if(!value.equals(1.0)) {
				return false;
			}
		}
		return true;
	}
	
	public static boolean isMaxMinTransitive(IFuzzySet relation) {
		if (!isUtimesURelation(relation)) {
			throw new IllegalArgumentException("Argument isn't binary relation");
		}
		IDomain domain = relation.getDomain();
		int n = (int) Math.sqrt(domain.getCardinality());
		for(int x = 0; x < n; ++x) {
			for(int z = 0; z < n; ++z) {
				Double miXZ = relation.getValueAt(standard_index(domain, x, z));
				List<Double> minValues = new ArrayList<>();
				for(int y = 0; y < n; ++y) {
					if(y == z || y == z) continue;
					Double miXY = relation.getValueAt(standard_index(domain, x, y));
					Double miYZ = relation.getValueAt(standard_index(domain, y, z));
					double min = Double.min(miXY, miYZ);
					minValues.add(min);
				}
				Double d = Collections.max(minValues);
				if(miXZ < d) return false;
			}
		}
		return true;
	}
	
	public static IFuzzySet compositionOfBinaryRelations(IFuzzySet a, IFuzzySet b) {
		IDomain Y1 = a.getDomain().getComponent(1);
		IDomain Y2 = b.getDomain().getComponent(0);
		if(!Y1.equals(Y2)) {
			throw new IllegalArgumentException("Fuzzy sets' domain aren't paired");
		}
		
		return calcMaxMinProduct(a, b);
	}
	
	public static boolean isFuzzyEquivalence(IFuzzySet set) {
		boolean value = isReflexive(set) && isSymmetric(set) && isMaxMinTransitive(set);
		return value;
	}
	
}
