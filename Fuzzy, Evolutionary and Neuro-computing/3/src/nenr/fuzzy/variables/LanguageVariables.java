package nenr.fuzzy.variables;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

import nenr.fuzzy.utils.Utils;
import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;
import nenr.lab1.functions.Operations;
import nenr.lab1.fuzzy.CalculatedFuzzySet;
import nenr.lab1.fuzzy.IFuzzySet;
import nenr.lab1.fuzzy.MutableFuzzySet;
import nenr.lab1.fuzzy.StandardFuzzySets;

public class LanguageVariables {
	
	public static DistanceVars DISTANCE = new DistanceVars();
	public static VelocityVars VELOCITY = new VelocityVars();
	public static SVars GOAL = new SVars();
	public static AkcelVars AKCEL = new AkcelVars();
	public static DirectionVars DIRECTION = new DirectionVars();
	
	public static IFuzzySet any(List<IFuzzySet> list) {
		BiFunction<IFuzzySet, IFuzzySet, IFuzzySet> orFunction = (a,b) -> Operations.binaryOperation(a, b, Operations.zadehOr());
		List<IFuzzySet> xs = new ArrayList<>(list);
		IFuzzySet result = xs.get(0);
		xs.remove(0);
		for(IFuzzySet f : xs) {
			result = orFunction.apply(result, f);
		}
		return result;
	}
	
	public static IFuzzySet any(IFuzzySet ... fuzzy) {
		List<IFuzzySet> xs = Utils.toList(fuzzy);
		return LanguageVariables.any(xs);
	}
	
	public static class DistanceVars{
		public IFuzzySet CLOSE_LEFT;
		public IFuzzySet CLOSE_RIGHT;
		public IFuzzySet FAR;
		public List<IFuzzySet> vars;
		{
			IDomain domain = InputDomains.distance();
			CLOSE_LEFT  = new CalculatedFuzzySet(domain, StandardFuzzySets.IFunction(100, 400));
			FAR = new CalculatedFuzzySet(domain, StandardFuzzySets.lambdaFunction(200, 650, 800));
			CLOSE_RIGHT = new CalculatedFuzzySet(domain, StandardFuzzySets.gammaFunction(750, 1100));
			vars = Utils.toList(CLOSE_LEFT, CLOSE_RIGHT, FAR);
		}
		
	}
	
	public static class VelocityVars{
		public IFuzzySet SLOW;
		public IFuzzySet FAST;
		public List<IFuzzySet> vars;
		{
			//0 ,100
			IDomain domain = InputDomains.velocity();
			SLOW = new CalculatedFuzzySet(domain, StandardFuzzySets.IFunction(20, 50));
			FAST = new CalculatedFuzzySet(domain, StandardFuzzySets.gammaFunction(40, 60));
			vars = Utils.toList(SLOW, FAST);
		}
	}
	
	public static class SVars{
		public IFuzzySet GOOD;
		public IFuzzySet BAD;
		public List<IFuzzySet> vars;
		{
			IDomain domain = InputDomains.goal();
			BAD = new MutableFuzzySet(domain).set(DomainElement.of(0), 1.0);
			GOOD = new MutableFuzzySet(domain).set(DomainElement.of(1), 1.0);
			vars = Utils.toList(BAD, GOOD);
		}
	}
	
	public static class AkcelVars{
		public IFuzzySet SLOWDOWN;
		public IFuzzySet MAINTAIN;
		public IFuzzySet ACCELERATE;
		public List<IFuzzySet> vars;
		{
			// -100, 100
			IDomain domain = InputDomains.acc();
			SLOWDOWN = new CalculatedFuzzySet(domain, StandardFuzzySets.IFunction(-80, -10));
			MAINTAIN = new CalculatedFuzzySet(domain, StandardFuzzySets.lambdaFunction(-20, 0, 20));
			ACCELERATE = new CalculatedFuzzySet(domain, StandardFuzzySets.gammaFunction(-20, 50));
			vars = Utils.toList(SLOWDOWN, MAINTAIN, ACCELERATE);
		}
	}
	
	public static class DirectionVars{
		public IFuzzySet LEFT;
		public IFuzzySet MAINTAIN;
		public IFuzzySet RIGHT;
		public List<IFuzzySet> vars;
		{
			IDomain domain = InputDomains.korm();
			RIGHT     = new CalculatedFuzzySet(domain, StandardFuzzySets.IFunction(-60, -10));
			MAINTAIN = new CalculatedFuzzySet(domain, StandardFuzzySets.lambdaFunction(-15, 0, 15));
			LEFT    = new CalculatedFuzzySet(domain, StandardFuzzySets.gammaFunction(10, 60));
			vars = Utils.toList(LEFT, MAINTAIN, RIGHT);
		}
	}
}
