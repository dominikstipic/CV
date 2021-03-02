package hr.fer.zemris.java.hw07.commands;

import static hr.fer.zemris.java.hw07.utils.Util.relativize;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import java.util.Objects;

import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Removes whole content of given directory.
 * Argument is directory denoted with relative or apsoulthe path.
 * Deleted files cannot be retrived
 * @author Dominik Stipić
 *
 */
public class RmTree implements ShellCommand{

	
	/**
	 * Visits directory and removes its content
	 * @author Dominik Stipić
	 *
	 */
	private class ShellVisitor extends SimpleFileVisitor<Path>{
		@Override
		public FileVisitResult postVisitDirectory(Path path, IOException arg1) throws IOException {
			Files.delete(path);
			return FileVisitResult.CONTINUE;
		}

		@Override
		public FileVisitResult visitFile(Path path, BasicFileAttributes arg1) throws IOException {
			Files.delete(path);
			return FileVisitResult.CONTINUE;
		}
	}
	
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Rmtree command expects 1 argument");
			return ShellStatus.CONTINUE;
		}
		
		ShellParser parser = new ShellParser(arguments);
		Path dir = null;
		try {
			List<String> list = parser.process();
			if (list.size() != 1) {
				env.writeln("Rmtree command expects 1 argument");
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
		
		if(!Files.isDirectory(dir)) {
			env.writeln("argument must be directory");
			return ShellStatus.CONTINUE;
		}
		
		try {
			Files.walkFileTree(dir, new ShellVisitor());
		} catch (IOException e) {
			e.printStackTrace();
			env.writeln("Error occured while deleting directory");
		}
		
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "rmtree";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Removes given directory and his substructure from file system",
				"Expects one argument",
				"rmtree [DIR]"
		);
	}
	

}
