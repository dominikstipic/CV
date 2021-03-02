package hr.fer.zemris.lsystem.impl.commands;

import hr.fer.zemris.lsystem.impl.Command;
import hr.fer.zemris.lsystem.impl.Context;
import hr.fer.zemris.lsystem.impl.TurtleState;
import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.math.Vector2D;

/**
 * Rotates turtle by certain degree
 * @author Dominik Stipic
 *
 */
public class RotateCommand implements Command {
	private double angle;
	
	/**
	 * Constructor with angle for turtle's rotation
	 * @param angle - rotation angle in degrees
	 */
	public RotateCommand(double angle) {
		this.angle = angle;
	}

	@Override
	public void execute(Context ctx, Painter painter) {
		TurtleState state = ctx.getCurrentState();
		Vector2D direction = state.getDirection();
		direction.rotate(angle);

	}

}
