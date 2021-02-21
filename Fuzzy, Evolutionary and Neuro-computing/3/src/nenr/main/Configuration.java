package nenr.main;

import nenr.fuzzy.coders.DefuzzerFactory;
import nenr.fuzzy.coders.FuzzerFactory;
import nenr.fuzzy.coders.IDefuzzifier;
import nenr.fuzzy.coders.IFuzzifier;
import nenr.fuzzy.machine.IMachine;
import nenr.fuzzy.machine.MachineFactory;

public class Configuration {
	private static IDefuzzifier defuzzer;
	private static IFuzzifier fuzzifier;
	private static IMachine machine;
	
	static{
		defuzzer = DefuzzerFactory.COA;
		fuzzifier = FuzzerFactory.SINGLETON;
		machine = MachineFactory.minimumMachine();
	}

	public static IDefuzzifier getDefuzzer() {
		return defuzzer;
	}
	
	public static IFuzzifier getFuzzifier() {
		return fuzzifier;
	}

	public static IMachine getMachine() {
		return machine;
	}

	
	
	
	
	
}
