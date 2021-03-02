package hr.fer.zemris.lsystem.impl.commands;

import hr.fer.zemris.lsystem.impl.Command;
import hr.fer.zemris.lsystem.impl.Context;
import hr.fer.zemris.lsystems.Painter;

/**
 * Pops current state from context ObjectStack
 * @author Dominik Stipic
 *
 */
public class PopCommand implements Command{

	@Override
	public void execute(Context ctx, Painter painter) {
		ctx.popState();
	}
	
}
