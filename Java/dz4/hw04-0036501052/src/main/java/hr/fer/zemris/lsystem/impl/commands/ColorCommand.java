package hr.fer.zemris.lsystem.impl.commands;

import java.util.Objects;
import java.awt.Color;


import hr.fer.zemris.lsystem.impl.Command;
import hr.fer.zemris.lsystem.impl.Context;
import hr.fer.zemris.lsystem.impl.TurtleState;
import hr.fer.zemris.lsystems.Painter;

/**
 * Sets color of fractal line
 * @author Dominik Stipic
 *
 */
public class ColorCommand implements Command {
	private Color color;
	
	/**
	 * Constructor which creates this command 
	 * @param color with which fractal will be colored
	 */
	public ColorCommand(Color color) {
		Objects.requireNonNull(color);
		this.color = color;
	}
	

	@Override
	public void execute(Context ctx, Painter painter) {
		TurtleState state = ctx.getCurrentState();
		state.setColor(color);
	}

}
