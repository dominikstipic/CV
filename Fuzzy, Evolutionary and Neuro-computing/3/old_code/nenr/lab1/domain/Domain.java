package nenr.lab1.domain;

import java.util.Iterator;

import nenr.fuzzy.utils.Utils;

public abstract class Domain implements IDomain{
	
	@Override
	public abstract int getCardinality();

	@Override
	public abstract IDomain getComponent(int component);

	@Override
	public abstract int getNumberOfComponents();

	@Override
	public abstract int IndexOfElement(DomainElement elem);

	@Override
	public abstract DomainElement elementForIndex(int index);

	public static IDomain intRange(int lower, int upper) {
		SimpleDomain domain = new SimpleDomain(lower, upper);
		return domain;
	}
	
	private static SimpleDomain[] convertToSimple(IDomain domain) {
		int n = domain.getNumberOfComponents();
		SimpleDomain[] simpleDomains = new SimpleDomain[n];
		for(int i = 0; i < n; ++i) {
			simpleDomains[i] = (SimpleDomain) domain.getComponent(i);
		}
		return simpleDomains;
	}
	
	public static Domain combine(IDomain a, IDomain b) {
		SimpleDomain[] first, second;
		if(!SimpleDomain.isSimple(a)) {
			first = convertToSimple(a);
		}
		else {
			first = new SimpleDomain[] {(SimpleDomain)a};
		}
		if(!SimpleDomain.isSimple(b)) {
			second = convertToSimple(b);
		}
		else {
			second = new SimpleDomain[] {(SimpleDomain)b};
		}
		SimpleDomain[] arr = Utils.append(first, second);
		CompositeDomain d = new CompositeDomain(arr);
		return d;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		
		IDomain otherDomain = (IDomain) obj;
		if(otherDomain.getCardinality() != getCardinality()) return false;
		else if(otherDomain.getNumberOfComponents() != getNumberOfComponents()) return false;
		
		Iterator<DomainElement> iter = this.iterator();
		Iterator<DomainElement> iterOther = otherDomain.iterator();
		
		while(iter.hasNext() && iterOther.hasNext()) {
			DomainElement d1 = iter.next();
			DomainElement d2 = iterOther.next();
			if(!d1.equals(d2)) return false;
		}
		return true;
	}
	
}
