package nenr.fuzzy.database;

import static nenr.fuzzy.utils.Utils.toList;
import static nenr.fuzzy.variables.LanguageVariables.DIRECTION;
import static nenr.fuzzy.variables.LanguageVariables.DISTANCE;
import static nenr.fuzzy.variables.LanguageVariables.GOAL;
import static nenr.fuzzy.variables.LanguageVariables.VELOCITY;
import static nenr.fuzzy.variables.LanguageVariables.any;

import java.util.List;

import nenr.lab1.fuzzy.IFuzzySet;

public class KormiloDatabaseFactory extends DatabaseFactory{

	@Override
	protected List<List<IFuzzySet>> getAntecenents(){
		List<IFuzzySet> velocity = VELOCITY.vars;
		List<IFuzzySet> goal = DIRECTION.vars;
		List<List<IFuzzySet>> result = List.of(
				toList(DISTANCE.CLOSE_LEFT, any(velocity), any(goal)),  
				toList(DISTANCE.CLOSE_RIGHT, any(velocity), any(goal)),  
				toList(DISTANCE.FAR, any(velocity), GOAL.GOOD),
				toList(DISTANCE.FAR, any(velocity), GOAL.BAD));
		return result;
	}
	
	@Override
	protected List<IFuzzySet> getConsequences(){
		return toList(DIRECTION.RIGHT, DIRECTION.LEFT, DIRECTION.MAINTAIN, DIRECTION.LEFT);
	}
	
}
