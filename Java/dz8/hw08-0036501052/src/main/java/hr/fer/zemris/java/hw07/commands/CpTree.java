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
 * Copies directory subtree in other directory.
 * if Provided destination directory doesn't exist,command 
 * renames source directories into appropriate name and copies directory subtree to 
 * destination.
 * @author Dominik Stipić
 *
 */
public class CpTree implements ShellCommand{

	
	/**
	 * Visits provided files and writes their name in given format.
	 * @author Dominik Stipić
	 *
	 */
	private class ShellVisitor extends SimpleFileVisitor<Path>{
		Path dest;
		
		public ShellVisitor(Path dest) {
			this.dest = dest;
		}

		@Override
		public FileVisitResult postVisitDirectory(Path path, IOException arg1) throws IOException {
			dest = dest.getParent();
			return FileVisitResult.CONTINUE;
		}

		@Override
		public FileVisitResult preVisitDirectory(Path path, BasicFileAttributes arg1) throws IOException {
			dest = Files.createDirectories(Paths.get(dest.toString(), path.getFileName().toString()));
			return FileVisitResult.CONTINUE;
		}

		@Override
		public FileVisitResult visitFile(Path path, BasicFileAttributes arg1) throws IOException {
			Path file = Paths.get(dest.toString(), path.getFileName().toString());
			Files.copy(path, file);
			return FileVisitResult.CONTINUE;
		}

	}
	
	
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("CpTree command expects 2 arguments");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		Path dest = null;
		Path source = null;
		try {
			List<String> list = parser.process();
			if(list.size() != 2) {
				env.writeln("Cptree command excepts 2 arguments");
				return ShellStatus.CONTINUE;
			}
			source = Paths.get(list.get(0));
			dest = Paths.get(list.get(1));
		} catch (Exception e1) {
			env.writeln("error occured while parsing input - " + e1.getMessage());
			return ShellStatus.CONTINUE;
		}
		
		source = relativize(env.getCurrentDirectory(), source);
		dest = relativize(env.getCurrentDirectory(), dest);
		if(dest == null || source == null ) {
			env.writeln("Provided arguments cannot be interpreted as valid path - " + dest +"," + source);
			return ShellStatus.CONTINUE;
		}
		if(Files.notExists(source) || !Files.isDirectory(source)) {
			env.writeln("Provided source path cannot be found -> "  + source);
			return ShellStatus.CONTINUE;
		}
		if(Files.exists(dest)) {
			ShellVisitor visitor = new ShellVisitor(dest);
			try {
				Files.walkFileTree(source, visitor);
			} catch (Exception e) {
				env.writeln("Error while visiting subfiles -> "  + e.getMessage());
			}
		}
		return ShellStatus.CONTINUE;
		
	}

	@Override
	public String getCommandName() {
		return "cptree";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Copies directory substructure to the new destination.",
				"Takes two arguments",
				"cptree [SOURCE][DESTINATION]"
			);
	}

	
}
