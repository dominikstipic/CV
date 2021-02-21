package nenr.fuzzy.database;

import static nenr.fuzzy.utils.Utils.toList;
import static nenr.fuzzy.variables.LanguageVariables.AKCEL;
import static nenr.fuzzy.variables.LanguageVariables.DIRECTION;
import static nenr.fuzzy.variables.LanguageVariables.DISTANCE;
import static nenr.fuzzy.variables.LanguageVariables.GOAL;
import static nenr.fuzzy.variables.LanguageVariables.VELOCITY;
import static nenr.fuzzy.variables.LanguageVariables.any;

import java.util.List;

import nenr.lab1.fuzzy.IFuzzySet;


public class AkcelDatabaseFactory extends DatabaseFactory{
	
	@Override
	protected List<List<IFuzzySet>> getAntecenents(){
		List<IFuzzySet> distance = DISTANCE.vars;
		List<IFuzzySet> goal = DIRECTION.vars;
		IFuzzySet CLOSE = any(DISTANCE.CLOSE_LEFT, DISTANCE.CLOSE_RIGHT);
		
		List<List<IFuzzySet>> result = List.of(
				toList(CLOSE, VELOCITY.FAST, any(goal)),
				toList(DISTANCE.FAR, VELOCITY.SLOW, GOAL.GOOD),
				toList(any(distance), VELOCITY.FAST, GOAL.BAD),
				toList(DISTANCE.FAR, VELOCITY.FAST, GOAL.GOOD));
		return result;
	}
	
	@Override
	protected List<IFuzzySet> getConsequences(){
		return toList(AKCEL.SLOWDOWN, AKCEL.ACCELERATE, AKCEL.SLOWDOWN, AKCEL.MAINTAIN);
	}
	
	

	
}
