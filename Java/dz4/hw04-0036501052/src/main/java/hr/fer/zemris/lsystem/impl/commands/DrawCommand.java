package hr.fer.zemris.lsystem.impl.commands;

import hr.fer.zemris.lsystem.impl.Command;
import hr.fer.zemris.lsystem.impl.Context;
import hr.fer.zemris.lsystem.impl.TurtleState;
import hr.fer.zemris.lsystems.Painter;
import hr.fer.zemris.math.Vector2D;

/**
 * Drawes fractal lines to provided painter
 * @author DOminik Stipic
 *
 */
public class DrawCommand implements Command{
	private double step;
	/**
	 * Thinckness of lines which turtle is drawing
	 */
	private static final int LINE_THICKNESS = 2;
	
	/**
	 * Constructor with step variable which scales turtle unit length
	 * @param step - scales unit length
	 */
	public DrawCommand(double step) {
		this.step = step;
	}
	


	@Override
	public void execute(Context ctx, Painter painter) {
		TurtleState state = ctx.getCurrentState();
		Vector2D startPosition = state.getPosition();
		Vector2D offset = state.getDirection().scaled(step * state.getUnitLength());
		Vector2D endPosition = startPosition.translated(offset);
		
		painter.drawLine(startPosition.getX(), startPosition.getY(), endPosition.getX(), endPosition.getY(), state.getColor(), LINE_THICKNESS);
		state.setPosition(endPosition);
	}

}
