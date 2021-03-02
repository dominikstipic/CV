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
 * Prints whole directory subtree into shell.
 * Printed text is fomrated into as a tree,
 * Files which are indended the most in right are
 * deepest in tree structure.
 * @author Dominik Stipić
 *
 */
public class Tree implements ShellCommand{
	
	/**
	 * Visits provided files and writes their name in given format.
	 * @author Dominik Stipić
	 *
	 */
	private class ShellVisitor extends SimpleFileVisitor<Path>{
		/**
		 * Shell Environmet
		 */
		private Environment env;
		/**
		 * level of indentation
		 */
		private int level = 0;
		
		/**
		 * Creates Visitor
		 * @param env - writes on this shell
		 */
		public ShellVisitor(Environment env) {
			Objects.requireNonNull(env);
			this.env = env;
		}
		
		@Override
		public FileVisitResult postVisitDirectory(Path path, IOException arg1) throws IOException {
			level -=2;
			return FileVisitResult.CONTINUE;
		}

		@Override
		public FileVisitResult preVisitDirectory(Path path, BasicFileAttributes arg1) throws IOException {
			String dirName = path.getFileName().toString();
			if(level == 0) {
				env.writeln(dirName);
			}
			else {
				env.writeln(String.format("%" + level + "s%s","",dirName));
			}
			level += 2;
			return FileVisitResult.CONTINUE;
		}

		@Override
		public FileVisitResult visitFile(Path path, BasicFileAttributes arg1) throws IOException {
			String fileName = path.getFileName().toString();
			env.writeln(String.format("%" + level + "s%s","",fileName));
			return FileVisitResult.CONTINUE;
		}

	}
	
	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Tree command expects 1 argument");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		
		Path path = null;
		try {
			List<String> list = parser.process();
			if(list.size() != 1) {
				env.writeln("Tree command excepts 1 argument");
				return ShellStatus.CONTINUE;
			}
			path = Paths.get(list.get(0));
		} catch (Exception e1) {
			env.writeln("error occured while parsing input - " + e1.getMessage());
			return ShellStatus.CONTINUE;
		}
		
		path = relativize(env.getCurrentDirectory(), path);
		if(path == null) {
			env.writeln("Provided argument cannot be interpreted as path - " + arguments);
			return ShellStatus.CONTINUE;
		}
		
		if(!Files.isDirectory(path)) {
			env.writeln("Provided path isn't directory - " + arguments);
			return ShellStatus.CONTINUE;
		}
		try {
			Files.walkFileTree(path, new ShellVisitor(env));
		} catch (Exception e) {
			env.writeln("error occured while printing directory tree " );
		}
		
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "tree";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Prints the tree substructure of given directoy",
				"tree [DIR]"
				);
	}
	
}
