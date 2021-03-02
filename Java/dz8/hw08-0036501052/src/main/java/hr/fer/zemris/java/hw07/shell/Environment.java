package hr.fer.zemris.java.hw07.shell;

import java.nio.file.Path;
import java.util.SortedMap;

/**
 * Represents information from {@link MyShell}.
 * Contians methods which can: write on, read from and change {@link MyShell}
 * state.
 * @author Dominik Stipić
 *
 */
public interface Environment {
	/**
	 * Reads a line from user.
	 * @return user input
	 * @throws ShellIOException - if errror occured while
	 * 	reading input 
	 */
	String readLine() throws ShellIOException;
	
	/**
	 * Writes a line witout transiting in a new row. 
	 * @param text which will be writen to standard output.
	 * @throws ShellIOException if error occurs during writing on
	 * standard output
	 */
	void write(String text) throws ShellIOException;
	
	/**
	 * Writes a line and transites in a new row
	 * @param text which will be writed to standard output.
	 * @throws ShellIOException if error occurs during writing on
	 * standard output
	 */
	void writeln(String text) throws ShellIOException;
	
	/**
	 * Returns all supported shell commands.
	 * @return all available commands
	 */
	SortedMap<String, ShellCommand> commands();
	
	/**
	 * Returns multiline symbols
	 * @return currently used multiline symbol
	 */
	Character getMultilineSymbol();
	
	/**
	 * Sets shell multiline symbol 
	 * @param symbol - new shell symbol
	 */
	void setMultilineSymbol(Character symbol);
	
	/**
	 * Gets shell prompt symbol
	 * @return shell -  prompt symbol
	 */
	Character getPromptSymbol();
	
	/**
	 * Sets shell prompt symbol
	 * @param symbol -  new shell prompt symbol
	 */
	void setPromptSymbol(Character symbol);
	
	/**
	 * Gets shell moreliness symbol
	 * @return shell moreliness symbol
	 */
	Character getMorelinesSymbol();
	
	/**
	 * Sets shell morelines symbol
	 * @param symbol new shell moreliness symbol
	 */
	void setMorelinesSymbol(Character symbol);
	
	/**
	 * Gets current directory in which user is positioned
	 * @return path of current directory
	 */
	Path getCurrentDirectory();
	
	/**
	 * Sets current shell directory into given one
	 * @param path of new directory
	 */
	void setCurrentDirectory(Path path);
	
	/**
	 * Gets the data from shared data space
	 * @param key of requesteddata 
	 * @return requested data
	 */
	Object getSharedData(String key);
	
	/**
	 * Sets the data in shared data space
	 * @param key of data
	 * @param value data which is going to be saved in data space
	 */
	void setSharedData(String key, Object value);
}
