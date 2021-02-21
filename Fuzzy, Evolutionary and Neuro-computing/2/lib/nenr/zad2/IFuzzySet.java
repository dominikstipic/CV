package nenr.zad2;

import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;

public interface IFuzzySet {
	IDomain getDomain();
	double getValueAt(DomainElement element);
}
	