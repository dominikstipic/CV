package hr.fer.zemris.java.hw07.commands;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributeView;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.sql.Date;
import java.text.SimpleDateFormat;
import java.util.List;
import java.util.Objects;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;
import static hr.fer.zemris.java.hw07.utils.Util.relativize;

/**
 * Lists content of directory.
 * Also prints detail information about file atributtes.
 * @author Dominik Stipić
 *
 */
public class Ls implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Ls command expects 1 argument");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		
		Path dir = null;
		try {
			dir = Paths.get(parser.process().get(0));
		} catch (Exception e1) {
			env.writeln("error occured while parsing input - " + e1.getMessage());
			return ShellStatus.CONTINUE;
		}
		
		dir = relativize(env.getCurrentDirectory(), dir);
		if(dir == null) {
			env.writeln("Provided argument cannot be interpreted as path - " + arguments);
			return ShellStatus.CONTINUE;
		}
		
		if(!Files.isDirectory(dir) ) {
			env.writeln("file with given path isn't directory - " + arguments);
			return ShellStatus.CONTINUE;
		}
		try {
			for(File f:dir.toFile().listFiles()) {
				printAttributes(env, f.toPath());
			}
		} catch (Exception e) {
			env.writeln("error occured during reading file atributes from - " + dir);
		}
		return ShellStatus.CONTINUE;
	}
	
	
	/**
	 * Prints file attributes on shell
	 * @param env - shell on which is going to be written
	 * @param path - file path
	 * @throws IOException - if ioexception occurs
	 */
	private void printAttributes(Environment env ,Path path) throws IOException {
		String [] atribs = new String []{"-", "-", "-", "-"};
		if(Files.isDirectory(path)) {
			atribs[0] = "d";
		}
		if(Files.isReadable(path)) {
			atribs[1] = "r";
		}
		if(Files.isWritable(path)) {
			atribs[2] = "w";
		}
		if(Files.isExecutable(path)) {
			atribs[3] = "x";
		}
		
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		BasicFileAttributeView faView = Files.getFileAttributeView(path, BasicFileAttributeView.class, LinkOption.NOFOLLOW_LINKS);
		BasicFileAttributes attributes = faView.readAttributes();
		FileTime fileTime = attributes.creationTime();
		String formattedDateTime = sdf.format(new Date(fileTime.toMillis()));

		for(String s:atribs) {
			env.write(s + " ");
		}
		env.write(String.format("%10d ",Files.size(path)));
		env.write(formattedDateTime+" ");
		env.writeln(path.getFileName().toString());
	}
	
	@Override
	public String getCommandName() {
		return "ls";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Lists directory content.",
				"List information about the files attributes",
				"ls [FILE]"
		);
	}
	
}
