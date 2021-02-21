package nenr.zad3;

import nenr.zad1.DomainElement;
import nenr.zad1.IDomain;
import nenr.zad2.IFuzzySet;
import nenr.zad2.MutableFuzzySet;

public class Operations {

	public static IFuzzySet unaryOperation(IFuzzySet set, IUnaryFunction function) {
		IDomain domain = set.getDomain();
		MutableFuzzySet resultSet = new MutableFuzzySet(domain);
		for(DomainElement elem : domain) {
			double old_value = set.getValueAt(elem);
			double new_value = function.valueAt(old_value);
			resultSet.set(elem, new_value);
		}
		return resultSet;
	};
	
	public static IFuzzySet binaryOperation(IFuzzySet a, IFuzzySet b, IBinaryFunction function) {
		if(! a.getDomain().equals(b.getDomain())) {
			throw new IllegalArgumentException("Fuzzy sets' domains should be equal");
		}
		IDomain domain = a.getDomain();
		MutableFuzzySet resultSet = new MutableFuzzySet(domain);
		for(DomainElement elem : domain) {
			double miA = a.getValueAt(elem);
			double miB = b.getValueAt(elem);
			double new_value = function.valueAt(miA, miB);
			resultSet.set(elem, new_value);
		}
		return resultSet;
	};

	public static IUnaryFunction zadehNot() {
		IUnaryFunction function = x -> 1-x;
		return function;
	};

	public static IBinaryFunction zadehAnd() {
		IBinaryFunction function = (x,y) -> Math.min(x, y);
		return function;
	};
	
	public static IBinaryFunction zadehOr() {
		IBinaryFunction function = (x,y) -> Math.max(x, y);
		return function;
	};
	
	public static IBinaryFunction hamacherTNorm(double param) {
		IBinaryFunction function = (a,b) -> (double) (a*b)/(param+(1-param)*(a+b-a*b));
		return function;
	};
	
	public static IBinaryFunction hamacherSNorm(double param) {
		IBinaryFunction function = (a,b) -> (double) (a+b-(2-param)*a*b)/(1-(1-param)*a*b);
		return function;
	};

}
