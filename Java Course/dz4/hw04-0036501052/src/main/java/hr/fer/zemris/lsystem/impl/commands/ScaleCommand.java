package hr.fer.zemris.lsystem.impl.commands;

import hr.fer.zemris.lsystem.impl.Command;
import hr.fer.zemris.lsystem.impl.Context;
import hr.fer.zemris.lsystem.impl.TurtleState;
import hr.fer.zemris.lsystems.Painter;

/**
 * Scales unit length by certain scaler 
 * @author Dominik Stipic
 *
 */
public class ScaleCommand implements Command {
	private double factor;
	
	/**
	 * Constructor with scaling factor
	 * @param factor - scaler for scaling turtle's unit length
	 */
	public ScaleCommand (double factor) {
		this.factor = factor;
	}
	
	@Override
	public void execute(Context ctx, Painter painter) {
		TurtleState state = ctx.getCurrentState();
		double unitLength = state.getUnitLength();
		state.setShift(unitLength * factor);
	}

}
