package hr.fer.zemris.java.hw07.commands;

import java.util.List;

import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Writes apsoulte path of current directory into standard output
 * @author Dominik Stipić
 *
 */
public class Pwd implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		if(arguments != null) {
			env.writeln("pwd command require 0 arguments");
			return ShellStatus.CONTINUE;
		}
		
		env.writeln(env.getCurrentDirectory().toAbsolutePath().normalize().toString());
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "pwd";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Prints current directory absolute path",
				"It reqires 0 arguments.",
				"pwd []"
		);
	}
}
