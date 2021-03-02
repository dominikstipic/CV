package hr.fer.zemris.java.hw07.commands;

import java.io.BufferedInputStream;
import java.nio.charset.Charset;
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
 * Interprets file according to given code page.
 * Then prints file content on standard output.	
 * Usage description:
 *<b>cat [FILE][CHARSET]</b>
 * @author Dominik Stipić
 */
public class Cat implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Cat command expects 1 or 2 arguments");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		Charset charset = Charset.defaultCharset();
		Path path = null;
		
		try {
			List<String> list = parser.process();
			if(list.size() > 2) {
				env.writeln("Too much arguments provided - schould be 1 or 2 ");
				return ShellStatus.CONTINUE;
			}
			path = Paths.get(list.get(0));
			if(list.size() == 2) {
				try {
					charset = Charset.forName(list.get(1));
				} catch (Exception e) {
					env.writeln("Could not find Charset with given name - " + list.get(1));
					return ShellStatus.CONTINUE;
				}
			}
			
			BufferedInputStream reader  = new BufferedInputStream(Files.newInputStream(path));
			byte buff[] = new byte[1024];
			while(true) {
				int read = reader.read(buff);
				if(read == -1) break;
				String line = new String(buff,charset);
				env.writeln(line);
			}
		} catch (Exception e1) {
			env.writeln("error occured while processing input -> " + e1.getMessage());
		}
		return ShellStatus.CONTINUE;
	}

	@Override
	public String getCommandName() {
		return "cat";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Interprets file according to given code page.",
				"Then prints file content on standard output",
				"cat [FILE][CHARSET]"
		);
	}
	
}
