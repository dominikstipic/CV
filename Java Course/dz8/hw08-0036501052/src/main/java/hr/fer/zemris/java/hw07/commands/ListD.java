package hr.fer.zemris.java.hw07.commands;

import java.nio.file.Path;
import java.util.List;
import java.util.Stack;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Prints internal stack conntent on standard output.
 * The lastly added directoy will be printed first
 * @author Dominik Stipić
 *
 */
public class ListD implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		Object o = env.getSharedData("cdstack");
		if(o == null ) {
			env.writeln("internal stack is empty");
			return ShellStatus.CONTINUE;
		}
		
		@SuppressWarnings("unchecked")
		Stack<Path> stack = (Stack<Path>) o;
		if(stack.isEmpty()) {
			env.writeln("There isn't stored directories");
			return ShellStatus.CONTINUE;
		}
		stack.forEach(dir -> env.writeln(dir.toString()));
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "listd";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Lists content form internal stack",
				"Expects zero argument",
				"listd []"
		);
	}

}
