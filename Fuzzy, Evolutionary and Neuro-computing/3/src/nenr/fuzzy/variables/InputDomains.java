package nenr.fuzzy.variables;

import nenr.lab1.domain.IDomain;
import nenr.lab1.domain.SimpleDomain;

public class InputDomains {
	public static final int[] DISTANCE = {-650, 650};
	public static final int[] VELOCITY = {0, 100};
	public static final int[] GOAL = {0, 1};
	public static final int[] ACC = {-100,100};
	public static final int[] KORM = {-90,90};
	
	public static IDomain distance() {
		return SimpleDomain.intRange(DISTANCE[0], DISTANCE[1]);
	}
	
	public static IDomain velocity() {
		return SimpleDomain.intRange(VELOCITY[0], VELOCITY[1]);
	}
	
	public static IDomain goal() {
		return SimpleDomain.intRange(GOAL[0], GOAL[1]);
	}
	
	public static IDomain acc() {
		return SimpleDomain.intRange(ACC[0], ACC[1]);
	}
	
	public static IDomain korm() {
		return SimpleDomain.intRange(KORM[0], KORM[1]);
	}
}
