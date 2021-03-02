package hr.fer.zemris.java.hw07.commands;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.Function;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Provides information about currenty used shell symbol for :PROMPT,
 * MORELINE and MULTILINE.
 * Also default symbols can be changed if provided appropriate arguments
 * @author Dominik Stipić
 *
 */
public class Symbol implements ShellCommand{

	/**
	 * Map which connects user input with corresponding function which
	 * gets searched symbol
	 */
	private Map <String, Function<Environment,Character>> getSymbol;
	/**
	 * Map which connects user input with corresponding  consumer which
	 * sets new symbol
	 */
	private Map <String, BiConsumer<Environment,Character>> changeSymbol;
	{
		getSymbol =  Map.of(
				"PROMPT", env -> env.getPromptSymbol(), 
				"MORELINES", env -> env.getMorelinesSymbol(), 
				"MULTILINE", env -> env.getMultilineSymbol());
		
		changeSymbol =  Map.of(
				"PROMPT", (env,sym) -> env.setPromptSymbol(sym), 
				"MORELINES", (env,sym) -> env.setMorelinesSymbol(sym), 
				"MULTILINE", (env,sym) -> env.setMultilineSymbol(sym));
	}
	
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Symbol command expects 1 or 2 arguments");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		
		try {
			List<String> list = parser.process();
			if(!(list.size() == 1 || list.size() == 2)) {
				env.writeln("Symbol command excepts 1 or 2 arguments");
				return ShellStatus.CONTINUE;
			}
			if(!getSymbol.containsKey(list.get(0))) {
				env.writeln("Provided symbol type does not exist - " + list.get(0));
				return ShellStatus.CONTINUE;
			}
			if(list.size() == 1) {
				env.writeln(String.format("Symbol for %s is '%s'", list.get(0), getSymbol.get(list.get(0)).apply(env)));
				return ShellStatus.CONTINUE;
			}
			if(list.get(1).length() != 1) {
				env.writeln("Shell symbols can only be characters, you provided string "+ list.get(1));
				return ShellStatus.CONTINUE;
			}
			Character newSymbol = list.get(1).charAt(0);
			Character oldSymbol = getSymbol.get(list.get(0)).apply(env);
			changeSymbol.get(list.get(0)).accept(env, newSymbol);
			env.writeln(String.format("Symbol for %s changed from '%s' to '%s'", list.get(0), oldSymbol, newSymbol));
		} catch (Exception e) {
			env.writeln("Error occured while parsing argument line - " + arguments);
		}
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "symbol";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Provides information about used shell symbols.",
				"Also provides option to change shell symbols",
				"symbol [SYM_TYPE][SYM]?"
		);
	}

}
