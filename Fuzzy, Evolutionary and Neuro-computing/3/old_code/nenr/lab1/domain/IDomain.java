package nenr.lab1.domain;

public interface IDomain extends Iterable<DomainElement>{
	int getCardinality();
	IDomain getComponent(int component);
	int getNumberOfComponents();
	int IndexOfElement(DomainElement elem);
	DomainElement elementForIndex(int index);
}
