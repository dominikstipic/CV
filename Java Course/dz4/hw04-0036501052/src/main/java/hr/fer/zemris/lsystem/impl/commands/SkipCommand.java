package hr.fer.zemris.lsystem.impl.commands;

import hr.fer.zemris.lsystem.impl.Command;
import hr.fer.zemris.lsystem.impl.Context;
import hr.fer.zemris.lsystem.impl.TurtleState;
import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.math.Vector2D;

/**
 * Moves turtle in her space without drawing to painter
 * @author Dominik Stipic
 *
 */
public class SkipCommand implements Command {
	private double step;
	
	/**
	 * Constructor with number of steps which turtle is going to make
	 * @param step - scaling factor for unit length 
	 */
	public SkipCommand(double step) {
		this.step = step;
	}

	@Override
	public void execute(Context ctx, Painter painter) {
		TurtleState state = ctx.getCurrentState();
		Vector2D oldVec = state.getPosition();
		double shift = state.getUnitLength();
		Vector2D newVec = oldVec.translated(oldVec).scaled(step * shift);
		
		state.setPosition(newVec);
	}

}
