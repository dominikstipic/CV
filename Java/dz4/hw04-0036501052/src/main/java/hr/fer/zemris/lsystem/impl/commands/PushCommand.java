package hr.fer.zemris.lsystem.impl.commands;

import hr.fer.zemris.lsystem.impl.Command;
import hr.fer.zemris.lsystem.impl.Context;
import hr.fer.zemris.lsystem.impl.TurtleState;
import hr.fer.zemris.lsystems.Painter;

/**
 * Pushes newly created current state to Context ObjectStack.
 * Used when turtle must remember her state
 * @author Dominik Stipic
 *
 */
public class PushCommand implements Command {

	@Override
	public void execute(Context ctx, Painter painter) {
		TurtleState state = ctx.getCurrentState();
		TurtleState newState = state.copy();
		ctx.pushState(newState);
	}

}
