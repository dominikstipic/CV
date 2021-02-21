package nenr.zad2;

public class StandardFuzzySets {

	public static IIntUnaryFunction IFunction(int alpha, int beta) {
		IIntUnaryFunction function = x -> {
			if(x < alpha) return 1.;
			else if(alpha <= x && x < beta) return (double) (beta-x)/(beta-alpha);
			else return 0.;
		};
		return function;
	}
	
	public static IIntUnaryFunction gammaFunction(int alpha, int beta) {
		IIntUnaryFunction function = x -> {
			if(x < alpha) return 0.;
			else if(alpha <= x && x < beta) return (double) (x-alpha)/(beta-alpha);
			else return 1.;
		};
		return function;
	}
	
	public static IIntUnaryFunction lambdaFunction(int alpha, int beta, int gamma) {
		IIntUnaryFunction function = x -> {
			if(x < alpha) return 0.;
			else if(alpha <= x && x < beta) return (double) (x-alpha)/(beta-alpha);
			else if(beta <= x && x < gamma) return (double) (gamma-x)/(gamma-beta);
			else return 0.;
		};
		return function;
	}
	
}
