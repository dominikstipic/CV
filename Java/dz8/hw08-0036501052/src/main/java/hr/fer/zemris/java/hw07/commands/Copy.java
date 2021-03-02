package hr.fer.zemris.java.hw07.commands;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import static hr.fer.zemris.java.hw07.utils.Util.relativize;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Copies file content to given destination.
 * If the same file exists on destination path, 
 * program will ask user if it wants to overwrite old file.
 * If directory is provided as destiantion, file will be 
 * copied to that directory. 
 * @author Dominik Stipić
 *
 */
public class Copy implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Copy command expects 2 arguments - source filepath and destination filepath");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		
		Path source = null;
		Path dest = null;
		try {
			List<String> list = parser.process();
			if(list.size() != 2) {
				env.writeln("Copy command expects 2 arguments");
				return ShellStatus.CONTINUE;
			}
			source = Paths.get(list.get(0));
			dest = Paths.get(list.get(1));
		} catch (Exception e1) {
			env.writeln("error occured while parsing input -> " + e1.getMessage());
			return ShellStatus.CONTINUE;
		}
		
		source = relativize(env.getCurrentDirectory(), source);
		dest = env.getCurrentDirectory().resolve(dest);
		
		if(dest == null ) {
			env.writeln("Provided arguments cannot be interpreted as valid path - " + dest +"," + source);
			return ShellStatus.CONTINUE;
		}
		
		if( Files.notExists(source) || Files.isDirectory(source)) {
			env.writeln("Provided path isn't directory - " + source);
			return ShellStatus.CONTINUE;
		}
		if(Files.exists(dest) && Files.isRegularFile(dest)) {
			env.writeln("Destination file already exist.Do you want overwrite it? y/n");
			String answer = env.readLine();
			while(!answer.equals("y") && !answer.equals("n")) {
				env.writeln("answer with y/n !");
				answer = env.readLine();
			}
			if(answer.equals("n")) {
				System.out.println("File won't be overwritten");
				return ShellStatus.CONTINUE;
			}
			System.out.println("Overwritting old file - " + dest.getFileName());
		}
		if(Files.isDirectory(dest)) {
			dest = Paths.get(dest.toString(), source.getFileName().toString());
		}
		
		try{
			copyFile(source, dest);
		}catch(IOException e) {
			env.writeln("Error while opening files - " + dest + ", " + source);
			return ShellStatus.CONTINUE;
		}
		env.writeln("File sucessfully copied");
		return ShellStatus.CONTINUE;
	}

	/**
	 * Copies file from source to destination
	 * @param source source file which will be copied 
	 * @param dest file new destiantion
	 * @throws IOException - if error while reading file occurs
	 */
	private void copyFile(Path source, Path dest) throws IOException {
		try(BufferedInputStream is = new BufferedInputStream(Files.newInputStream(source));
				BufferedOutputStream os = new BufferedOutputStream(Files.newOutputStream(dest))){
				byte buff[] = new byte[1024];
				while(true) {
					int read = is.read(buff);
					if(read == -1) break;
					os.write(buff, 0, read);
				}
		}
	}
	
	@Override
	public String getCommandName() {
		return "copy";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Copies file to given destination.",
				"If destination is directory than it copies file content in it.",
				"If file with same name already exists, asks for permission to overwrite it",
				"copy [FILE_SOURCE][FILE_DESTINATION]"
			);
	}
	
}
