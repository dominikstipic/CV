package hr.fer.zemris.java.hw07.commands;

import java.nio.file.Path;
import java.util.List;
import java.util.Stack;

import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Deletes lastly added directory from shared data stack.
 * Deleted directory is delteted temporary and it cannot be retrieved.
 * @author Dominik Stipić
 *
 */
public class DropD implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		Object o = env.getSharedData("cdstack");
		if(o == null ) {
			env.writeln("internal stack is empty");
			return ShellStatus.CONTINUE;
		}
		
		@SuppressWarnings("unchecked")
		Stack<Path> stack = (Stack<Path>) o;
		if(stack.size() == 0) {
			env.writeln("internal stack is empty");
			return ShellStatus.CONTINUE;
		}
		
		stack.pop();
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "dropd";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"drops lastly added directory path to internal stack",
				"Expects zero argument",
				"dropd []"
		);
	}

}
