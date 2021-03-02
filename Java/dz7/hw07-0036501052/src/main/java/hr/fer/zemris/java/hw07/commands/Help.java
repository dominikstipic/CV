package hr.fer.zemris.java.hw07.commands;

import java.util.List;
import java.util.Set;

import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * It Takes one or zero arguments. If no arguments are found,
 * it lists all available commands on this shell. If command name is
 * provided it prints that command description.
 * @author Dominik Stipić
 *
 */
public class Help implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		Set<String> commands = env.commands().keySet();
		if(arguments == null) {
			commands.forEach(c -> System.out.println(c));
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		
		try {
			List<String> list = parser.process();
			if(list.size() != 1) {
				env.writeln("Help command expects 1 or 0 arguments");
				return ShellStatus.CONTINUE;
			}
			ShellCommand command = env.commands().get(list.get(0));
			if(command == null) {
				env.writeln("Command not found - " + list.get(0));
				return ShellStatus.CONTINUE;
			}
			command.getCommandDescription().forEach(d -> System.out.println(d));
		} catch (Exception e) {
			env.writeln("Error occured while parsing argument line - " + arguments);
		}
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "help";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
			"Provides help information for Shell commands",
			"If started without arguments prints all available commands.",
			"If additionaly provided with command argument, prints detail description about that command",
			"help [command]?"
		);
	}

}
