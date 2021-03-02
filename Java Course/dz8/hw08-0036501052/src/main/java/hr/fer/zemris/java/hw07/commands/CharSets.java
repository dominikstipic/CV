package hr.fer.zemris.java.hw07.commands;

import java.nio.charset.Charset;
import java.util.List;
import java.util.SortedMap;

import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Lists all available code pages on this java platform.
 * @author Dominik Stipić
 *
 */
public class CharSets implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		if(arguments != null) {
			env.writeln("arguments for charsets aren't allowed, you provided - " + arguments);
			return ShellStatus.CONTINUE;
		}
		SortedMap<String, Charset> map = Charset.availableCharsets();
		map.forEach((k,v) -> env.writeln(k));
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "charsets";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Lists available code pages on this java platform",
				"charset []"
		);
	}
	
}
