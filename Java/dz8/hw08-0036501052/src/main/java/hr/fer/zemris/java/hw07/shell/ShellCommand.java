package hr.fer.zemris.java.hw07.shell;
import java.util.List;

/**
 * Models generic shell command.
 * It contains method which starts commands execution and methods which
 * provide additonal information about commands usage.
 * @author Dominik Stipić
 *
 */
public interface ShellCommand {
	/**
	 * Starts execution of command
	 * @param env Shell environment
	 * @param arguments argumetns of command
	 * @return Shell status which indicates if the shell is ready for termiantion 
	 * or not
	 */
	ShellStatus executeCommand(Environment env, String arguments);
	
	/**
	 * Information about shell commands name
	 * @return command name
	 */
	String getCommandName();
	
	/**
	 * Gives detail description about command usage and purpose
	 * @return list of Strings with command information.
	 */
	List<String> getCommandDescription();
}
