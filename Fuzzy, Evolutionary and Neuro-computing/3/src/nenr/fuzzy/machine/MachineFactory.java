package nenr.fuzzy.machine;

import java.util.function.BiFunction;
import nenr.lab1.functions.Operations;
import nenr.lab1.fuzzy.IFuzzySet;

public class MachineFactory {

	public static IMachine productMachine() {
		BiFunction<IFuzzySet, IFuzzySet, IFuzzySet> sNorm = (x,y) -> Operations.binaryOperation(x, y, Operations.zadehOr());
		BiFunction<Double, Double, Double> implication = (x, y) -> x*y;
		InferenceMachine machine = new InferenceMachine(sNorm, implication);
		return machine;
	}
	
	public static IMachine minimumMachine() {
		BiFunction<IFuzzySet, IFuzzySet, IFuzzySet> sNorm = (x,y) -> Operations.binaryOperation(x, y, Operations.zadehOr());
		BiFunction<Double, Double, Double> implication = (x, y) -> Math.min(x, y);
		InferenceMachine machine = new InferenceMachine(sNorm, implication);
		return machine;
	}
	
		
	
}
