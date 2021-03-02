package hr.fer.zemris.lsystem.impl;

import java.awt.Color;
import java.util.Objects;

import hr.fer.zemris.math.Vector2D;

/**
 * Encapsulates all data that represents turtle's state
 * @author Dominik Stipic 
 *
 */
public class TurtleState {
	/**
	 * Turtle position in coordinate grid
	 */
	private Vector2D position;
	/**
	 * Turtle position in coordinate grid
	 */
	private Vector2D direction;
	/**
	 * Color with which turtle is drawing
	 */
	private Color color;
	/**
	 * Length of turtle one step in watched direction 
	 */
	private double unitLength;
	
	/**
	 * Constructor for turtle state 
	 * @param position Turtle position in coordinate grid
 	 * @param direction Turtle position in coordinate grid
	 * @param color Color with which turtle is drawing
	 * @param unitLength Length of turtle one step in watched direction 
	 */
	public TurtleState(Vector2D position, Vector2D direction, Color color, double unitLength) {
		this.position = Objects.requireNonNull(position);
		Objects.requireNonNull(direction);
		direction.normalize();
		this.direction = direction;
		this.color = Objects.requireNonNull(color);
		this.unitLength = unitLength;
	}
	
	/**
	 * Copies all turtles state in newly created TurtleState
	 * @return copy of this TurtleState
	 */
	public TurtleState copy() {
		return new TurtleState(position.copy(), direction.copy(), color, unitLength);
	}

	/**
	 * Getter
	 * @return Turtle postion
	 */
	public Vector2D getPosition() {
		return position;
	}

	/**
	 * Getter
	 * @return turtle watched direction
	 */
	public Vector2D getDirection() {
		return direction;
	}

	/**
	 * Getter
	 * @return drawing color 
	 */
	public Color getColor() {
		return color;
	}

	/**
	 * Getter
	 * @return unit length of this turtle 
	 */
	public double getUnitLength() {
		return unitLength;
	}

	/**
	 * Setter 
	 * @param position Turtle postion
	 */
	public void setPosition(Vector2D position) {
		this.position = Objects.requireNonNull(position);
	}

	/**
	 * Setter
	 * @param direction - turtle watched direction
	 */
	public void setDirection(Vector2D direction) {
		this.direction = Objects.requireNonNull(direction);
	}

	/**
	 * Setter
	 * @param color - drawing color
	 */
	public void setColor(Color color) {
		this.color = Objects.requireNonNull(color);
	}

	/**
	 * Setter
	 * @param shift - unit length of this turtle 
	 */
	public void setShift(double shift) {
		this.unitLength = shift;
	}
	
	
	
}
