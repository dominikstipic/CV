package nenr.fuzzy.system;

import java.util.List;
import java.util.function.Function;
import nenr.fuzzy.variables.InputVariable;

public interface IFuzzySystem extends Function<List<InputVariable>, Double>{}
