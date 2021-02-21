package nenr.zad2;

import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;

public class MutableFuzzySet implements IFuzzySet{
	private double[] memberships;
	private IDomain domain;
	
	public MutableFuzzySet(IDomain domain) {
		this.domain = domain;
		memberships = new double[domain.getCardinality()];
	}

	@Override
	public IDomain getDomain() {
		return domain;
	}

	@Override
	public double getValueAt(DomainElement element) {
		int idx = domain.IndexOfElement(element);
		return memberships[idx];
	}
	
	public MutableFuzzySet set(DomainElement element, double value) {
		int idx = domain.IndexOfElement(element);
		memberships[idx] = value;
		return this;
	}
	
	
	
}
