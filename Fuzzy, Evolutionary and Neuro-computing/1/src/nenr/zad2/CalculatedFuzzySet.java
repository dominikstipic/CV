package nenr.zad2;

import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;

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
	

}
