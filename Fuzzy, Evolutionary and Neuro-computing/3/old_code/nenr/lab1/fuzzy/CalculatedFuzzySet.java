package nenr.lab1.fuzzy;

import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;

public class CalculatedFuzzySet implements IFuzzySet{
	private IIntUnaryFunction membershipFunction;
	private IDomain domain;
	
	public CalculatedFuzzySet(IDomain domain, IIntUnaryFunction membershipFunction) {
		this.membershipFunction = membershipFunction;
		this.domain = domain;
	}

	@Override
	public IDomain getDomain() {
		return domain;
	}

	@Override
	public double getValueAt(DomainElement element) {
		return membershipFunction.valueAt(domain.IndexOfElement(element));
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((domain == null) ? 0 : domain.hashCode());
		result = prime * result + ((membershipFunction == null) ? 0 : membershipFunction.hashCode());
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
		CalculatedFuzzySet other = (CalculatedFuzzySet) obj;
		if (domain == null) {
			if (other.domain != null)
				return false;
		} else if (!domain.equals(other.domain))
			return false;
		if (membershipFunction == null) {
			if (other.membershipFunction != null)
				return false;
		} else if (!membershipFunction.equals(other.membershipFunction))
			return false;
		return true;
	}
	
	

}
