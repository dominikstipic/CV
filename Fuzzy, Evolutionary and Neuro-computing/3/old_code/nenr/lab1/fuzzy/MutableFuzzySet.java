package nenr.lab1.fuzzy;

import java.util.Arrays;

import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;

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

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((domain == null) ? 0 : domain.hashCode());
		result = prime * result + Arrays.hashCode(memberships);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		MutableFuzzySet other = (MutableFuzzySet) obj;
		if (domain == null) {
			if (other.domain != null)
				return false;
		} else if (!domain.equals(other.domain))
			return false;
		if (!Arrays.equals(memberships, other.memberships))
			return false;
		return true;
	}
	
	
	
	
}
