package hr.fer.zemris.java.hw07.commands;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import java.util.Stack;
import static hr.fer.zemris.java.hw07.utils.Util.relativize;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Pushes current directory into shared stack.
 * When current directory is pushed,changes current working directory into
 * given one.
 * Expects one argument - new directory
 * @author Dominik Stipic
 *
 */
public class PushD implements ShellCommand {

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Pushd command expects 1 argument");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);

		Path path = null;
		try {
			List<String> list = parser.process();
			path = Paths.get(list.get(0));
			if(list.size() != 1) {
				env.writeln("Pushd command expects 1 argument -> " + arguments);
				return ShellStatus.CONTINUE;
			}
		} catch (Exception e1) {
			env.writeln("error occured while parsing input - " + e1.getMessage());
			return ShellStatus.CONTINUE;
		}
		
		path = relativize(env.getCurrentDirectory(), path);
		if(path == null) {
			env.writeln("Provided argument cannot be interpreted as valid path - " + arguments);
			return ShellStatus.CONTINUE;
		}
		
		@SuppressWarnings("unchecked")
		Stack<Path> stack = (Stack<Path>) env.getSharedData("cdstack");
		if(stack == null) {
			stack = new Stack<>();
			env.setSharedData("cdstack", stack);
		}
		stack.push(env.getCurrentDirectory());
		env.setSharedData("cdstack", stack);
		
		env.setCurrentDirectory(path);
		
		return null;
	}

	@Override
	public String getCommandName() {
		return "pushd";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Pushes current directory to the stack and sets current directory to given path",
				"Expects one argument",
				"pushd [DIR]"
		);
	}

}
