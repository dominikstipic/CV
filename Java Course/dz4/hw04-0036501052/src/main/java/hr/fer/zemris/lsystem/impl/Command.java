package hr.fer.zemris.lsystem.impl;

import hr.fer.zemris.lsystems.Painter;

/**
 * Interface which represents command for fractal creation
 * @author Dominik Stipic
 *
 */
public interface Command {
	/** 
	 * Starts the execution of command
	 * @param ctx - ObjactStact which stores current state of turtle 
	 * @param painter - Object which provides drawing capabilities
	 */
	void execute(Context ctx, Painter painter);
}
