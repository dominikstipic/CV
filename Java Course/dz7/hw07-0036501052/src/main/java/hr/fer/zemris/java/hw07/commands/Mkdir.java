package hr.fer.zemris.java.hw07.commands;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;

import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Creates directory or directories in specified directory path.
 * @author Dominik Stipić
 *
 */
public class Mkdir implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Mkdir command expects 1 argument");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		
		Path dest = null;
		try {
			List<String> list = parser.process();
			dest = Paths.get(list.get(0));
		} catch (Exception e1) {
			env.writeln("error occured while processing input -> " + e1.getMessage());
			return ShellStatus.CONTINUE;
		}
		
		try {
			Files.createDirectories(dest);
		} catch (IOException e) {
			env.writeln("Error occured while creating dirctory structure");
			return ShellStatus.CONTINUE;
		}
		env.writeln("successfuly created directories");
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "mkdir";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Creates the directories, if they do not already exist",
				"mkdir [PATH]"
		);
	}
	
}
