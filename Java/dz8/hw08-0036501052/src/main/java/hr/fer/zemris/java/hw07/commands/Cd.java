package hr.fer.zemris.java.hw07.commands;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;
import static hr.fer.zemris.java.hw07.utils.Util.relativize;

/**
 * Changes the shell working directory.
 * Command expects one argument:
 * relative or apsolute path of next working directory.
 * @author Dominik Stipic
 *
 */
public class Cd implements ShellCommand {

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Cd command expects 1 argument");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		Path dir = null;
		try {
			List<String> list = parser.process();
			if (list.size() != 1) {
				env.writeln("Cd command expects 1 argument");
				return ShellStatus.CONTINUE;
			}
			dir = Paths.get(list.get(0));
		} catch (Exception e) {
			env.writeln("error occured while parsing input -> " + e.getMessage());
			return ShellStatus.CONTINUE;
		}
		dir = relativize(env.getCurrentDirectory(), dir);
		if(dir == null) {
			env.writeln("Provided argument cannot be interpreted as valid path - " + arguments);
			return ShellStatus.CONTINUE;
		}
		
		env.setCurrentDirectory(dir);
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "cd";
	}
	
	@Override
	public List<String> getCommandDescription() {
		return List.of("Enables moving through directories.", "It reqires 1 argument which is directory name.",
				"cd [DIR]");
	}

}
