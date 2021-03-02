package hr.fer.zemris.java.custom.scripting.nodes;

/**
 * Class which represent simple processor with empty process method.
 * Processor must be "programmed" by user to do certain task.
 * @author Dominik Stipić
 *
 */
public class Processor {

	/**
	 * This method represents task which user must implement
	 * @param value - input value for this processor
	 */
	public void process(Object value) {}
}
