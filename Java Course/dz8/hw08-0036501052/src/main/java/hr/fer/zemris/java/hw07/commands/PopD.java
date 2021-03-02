package hr.fer.zemris.java.hw07.commands;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Stack;

import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Pops the directory from internal stack and sets that directory to current one.
 * @author Dominik Stipić
 *
 */
public class PopD implements ShellCommand{

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
		
		Path dir = stack.pop();
		if(Files.notExists(dir)) {
			env.writeln("poped directory doesn't exist");
			return ShellStatus.CONTINUE;
		}
		
		env.setCurrentDirectory(dir);
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "popd";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Pops stored path from internal stack and sets popped path to cuurent directoy",
				"Expects zero argument",
				"popd []"
		);
	}

}
