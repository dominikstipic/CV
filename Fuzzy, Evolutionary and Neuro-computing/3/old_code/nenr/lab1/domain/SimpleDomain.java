package nenr.lab1.domain;

import java.util.Iterator;
import java.util.NoSuchElementException;
import static java.lang.Math.abs;

public class SimpleDomain extends Domain{
	private int first;
	private int last;

	public SimpleDomain(int first, int last) {
		if (first > last){
			throw new IllegalArgumentException("Last bound cannot be bigger than first bound");
		}
		this.first = first;
		this.last = last;
	}
	
	public int getCardinality() {
		int x = first;
		int y = last;
		if(first < 0) {
			x = 0;
			y = last + abs(first); 
		}
		return y - x + 1;
	}
	
	@Override
	public IDomain getComponent(int component) {
		if(component > 1) {
			throw new IllegalArgumentException("Number of components in simple domain is always 1!");
		}
		return this;
	}

	@Override
	public int getNumberOfComponents() {
		return 1;
	}
	
	@Override
	public Iterator<DomainElement> iterator() {
		return new SimpleDomainIterator();
	}

	public int getFirst() {
		return first;
	}
	
	public int getLast() {
		return last;
	}
	
	@Override
	public int IndexOfElement(DomainElement elem) {
		if(elem.getNumberOfComponents() != 1) {
			throw new IllegalArgumentException("A domain doesn't contain the elements whose dimensionality surpasses 1");
		}
		int v = elem.getComponentValue(0);
		if(!(first <= v && v <= last)) {
			throw new IllegalArgumentException("A domain doesn't contain given element");
		}
		return v-first;
	}

	@Override
	public DomainElement elementForIndex(int index) {
		int values[] = new int[]{first+index};
		return new DomainElement(values);
	}
	
	public static boolean isSimple(IDomain domain) {
		int n = domain.getNumberOfComponents();
		return n == 1;
	}
	
	private class SimpleDomainIterator implements Iterator<DomainElement>{
		private int currentIdx = 0;
		private int n = SimpleDomain.this.getCardinality();
		
		@Override
		public boolean hasNext() {
			return currentIdx < n;
		}

		@Override
		public DomainElement next() {
			if(!hasNext()) {
				throw new NoSuchElementException("The iterator passed over all collection");
			}
			int element = SimpleDomain.this.first + currentIdx;
			++currentIdx;
			return new DomainElement(new int[] {element});
		}
		
	}

}
