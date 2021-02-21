package nenr.lab1.fuzzy;
import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;

public interface IFuzzySet {
	IDomain getDomain();
	double getValueAt(DomainElement element);
}
	