package nenr.lab1.domain;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

public class CompositeDomain extends Domain{
	private SimpleDomain[] domains;

	public CompositeDomain(SimpleDomain... domains) {
		this.domains = domains;
	}

	@Override
	public Iterator<DomainElement> iterator() {
		return new CompositeDomainIterator();
	}

	@Override
	public int getCardinality() {
		int n = 1;
		for(SimpleDomain d : domains) {
			n*= d.getCardinality();
		}
		return n;
	}

	@Override
	public IDomain getComponent(int component) {
		if(component >= getNumberOfComponents()) {
			throw new IllegalArgumentException("Number of components in simple domain is always 1!");
		}
		return domains[component];
	}

	@Override
	public int getNumberOfComponents() {
		return domains.length;
	}
	
	@Override
	public int IndexOfElement(DomainElement elem) {
		int i = 0;
		Iterator<DomainElement> it = iterator();
		while(it.hasNext()) {
			DomainElement de = it.next();
			if(de.equals(elem)) {
				return i;
			}
			++i;
		}
		throw new NoSuchElementException("Colection doesn't contains the given element");
	}
	
	@Override
	public DomainElement elementForIndex(int index) {
		if(index > getCardinality()) {
			throw new IndexOutOfBoundsException("Indexing out of bounds!");
		}
		int[] indices = decompose(index);
		int[] values  = new int[indices.length];
		for(int i = 0; i < indices.length; ++i) {
			int idx = indices[i];
			values[i] = this.domains[i].elementForIndex(idx).getComponentValue(0);
		}
		return new DomainElement(values);
	}
	
	
	public static IDomain fromSimple(IDomain ...domainArray) {
		if(domainArray.length <= 1) throw new IllegalArgumentException("Cannot create composite domain with only one simple domain");
		SimpleDomain[] domains = Arrays.asList(domainArray).stream().map(d -> (SimpleDomain) d).collect(Collectors.toList()).toArray(new SimpleDomain[0]);
		IDomain composite = CompositeDomain.combine(domains[0], domains[1]);
		for(int i = 2; i < domains.length; ++i) {
			SimpleDomain simple = domains[i];
			composite = CompositeDomain.combine(composite, simple);
		}
		return composite;
	}
	
	public static IDomain fromSimple(List<SimpleDomain> domains) {
		SimpleDomain[] arr = domains.toArray(new SimpleDomain[0]);
		return CompositeDomain.fromSimple(arr);
	}
	
	
	/**
	 * Decomposes simple array indexing to the matrix style indexing  
	 * @param n index in array style indexing
	 * @return matrix style indexing array (i,j)
	 */
	private int[] decompose(int n){
		int [] lengths = Arrays.asList(domains).stream().mapToInt(d -> d.getCardinality()).toArray();
		
		int[] weights = new int[lengths.length];
		Arrays.fill(weights, 1);
		Arrays.fill(weights, 1);
		for(int i = 0; i < lengths.length-1; ++i) {
			for(int j = i+1; j < lengths.length; ++j) {
				weights[i]*=lengths[j];
			}
		}
		
		// Standard matrix indices which first element is (0,0)
		int[] standard_idx = new int[weights.length];
		for(int i = 0; i  < weights.length; ++i) {
			for(int j = 0; j <= lengths[i]; ++j) {
				if(n - weights[i]*j < 0) {
					n -= weights[i]*(j-1);
					standard_idx[i] = j-1;
					break;
				}
				else if(n - weights[i]*j == 0) {
					n -= weights[i]*j;
					standard_idx[i] = j;
					break;
				}
			}
		}
		return standard_idx;
	}
	
	private class CompositeDomainIterator implements Iterator<DomainElement>{
		private int currentIdx = 0;
		
		@Override
		public boolean hasNext() {
			int n = getCardinality();
			return currentIdx < n;
		}

		@Override
		public DomainElement next() {
			if(!hasNext()) {
				throw new NoSuchElementException("The iterator passed over all collection");
			}
			DomainElement de = CompositeDomain.this.elementForIndex(currentIdx);
			++currentIdx; 
			return de;
		}
	}


}
