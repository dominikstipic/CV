package nenr.fuzzy.coders;

import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;

public class DefuzzerFactory {
	public static final IDefuzzifier COA = fuzzy -> {
		IDomain domain = fuzzy.getDomain();
		double sum1 = 0;
		double sum2 = 0;
		for(DomainElement el : domain) {
			double mi = fuzzy.getValueAt(el);
			double x  = el.getValues()[0];
			sum1 += x*mi;
			sum2 += mi;
		}
		if(Double.compare(sum2, 0.0) == 0) {
			return 0.;
		}
		else return (double) sum1/sum2;
	};
}
