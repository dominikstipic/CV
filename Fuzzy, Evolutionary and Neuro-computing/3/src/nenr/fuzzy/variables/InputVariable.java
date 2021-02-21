package nenr.fuzzy.variables;

import static nenr.fuzzy.variables.InputDomains.DISTANCE;
import static nenr.fuzzy.variables.InputDomains.VELOCITY;

import java.util.List;

import nenr.fuzzy.utils.Utils;
import nenr.lab1.domain.IDomain;

public class InputVariable {
	private int value;
	private IDomain domain;
	
	public InputVariable(int value, IDomain domain) {
		this.value = value;
		this.domain = domain;
	}
	public int getValue() {
		return value;
	}
	public void setValue(int value) {
		this.value = value;
	}
	public IDomain getDomain() {
		return domain;
	}
	public void setDomain(IDomain domain) {
		this.domain = domain;
	}
	@Override
	public String toString() {
		return "InputVariable [value=" + value + ", domain=" + domain + "]";
	}
	
	public static List<InputVariable> toVars(int L, int D, int LK, int DK, int V, int S){
		// [0,1300]
		int distance_left = (L + LK) / 2;
		int distance_right = (D + DK) / 2;
		//[-650,650]
		int distance = distance_left < distance_right ? L-1300/2 : 1300/2 - D;
		distance = (int) Utils.transformRange(distance, -650, 650, DISTANCE[0], DISTANCE[1]);
		// [0,100]
		int velocity = Math.min(V, 100); 
		velocity = (int) Utils.transformRange(velocity, 0, 100, VELOCITY[0], VELOCITY[1]);
		// [0,1]
		int goal = S;
		
		InputVariable distanceVariable = new InputVariable(distance, InputDomains.distance());
		InputVariable velocityVariable = new InputVariable(velocity, InputDomains.velocity());
		InputVariable goalVariable = new InputVariable(goal, InputDomains.goal());
		return List.of(distanceVariable,velocityVariable, goalVariable);
	} 
	
}
