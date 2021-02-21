package nenr.zad1;

import java.util.Iterator;

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
	
	public static Domain combine(IDomain a, IDomain b) {
		CompositeDomain d = new CompositeDomain(new SimpleDomain[]{(SimpleDomain)a,(SimpleDomain)b});
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
