package nenr.zad2.main;

import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;
import nenr.zad2.IFuzzySet;

public class Debug {
	public static void print(IDomain domain, String headingText) {
		if (headingText != null) {
			System.out.println(headingText);
		}
		for (DomainElement e : domain) {
			System.out.println("Element domene: " + e);
		}
		System.out.println("Kardinalitet domene je: " + domain.getCardinality());
		System.out.println();
	}
	
	public static void print(IFuzzySet fuzzy, String headingText) {
		if (headingText != null) {
			System.out.println(headingText);
		}
		for (DomainElement element : fuzzy.getDomain()) {
			double value = fuzzy.getValueAt(element);
			System.out.format("d(%s)=%f\n", element, value);
		}
		System.out.println();
	}
	
	
}
