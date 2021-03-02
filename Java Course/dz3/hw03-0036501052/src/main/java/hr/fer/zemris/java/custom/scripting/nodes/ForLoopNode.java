package hr.fer.zemris.java.custom.scripting.nodes;

import java.util.Objects;

import hr.fer.zemris.java.custom.scripting.elems.Element;
import hr.fer.zemris.java.custom.scripting.elems.ElementVariable;

public class ForLoopNode extends Node{
	private ElementVariable variable;
	private Element startExpression;
	private Element endExpression;
	private Element stepExpression;
	
	

	public ForLoopNode(ElementVariable variable, Element startExpression, Element endExpression,
			Element stepExpression) {
		this.variable = Objects.requireNonNull(variable);
		this.startExpression = Objects.requireNonNull(startExpression);
		this.endExpression = Objects.requireNonNull(endExpression);
		this.stepExpression = stepExpression;
	}

	public ElementVariable getVariable() {
		return variable;
	}
	

	public Element getStartExpression() {
		return startExpression;
	}

	public Element getEndExpression() {
		return endExpression;
	}

	public Element getStepExpression() {
		return stepExpression;
	}

	@Override
	public String toString() {
		if(stepExpression == null) {
			return "{$FOR " + variable.asText() + " " + startExpression.asText() + " " + endExpression.asText() + "$}";
		}
		else {
			return "{$FOR " + variable.asText() + " " + startExpression.asText() + " " + endExpression.asText() + " " + stepExpression.asText() + "$}";
		}
	}
	
	
}
