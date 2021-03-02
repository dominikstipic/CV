package hr.fer.zemris.java.hw07.commands;


import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import static hr.fer.zemris.java.hw07.utils.Util.relativize;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;

/**
 * Display file content in hexadecimal on standard output.
 * Expects file path as an argument.
 * @author Dominik Stipić
 *
 */
public class Hexdump implements ShellCommand{

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("Hexdump command expects 1 arguments");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		Path path = null;
		try {
			List<String> list = parser.process();
			if(list.size() != 1) {
				env.writeln("Hexdump command expects 1 arguments");
				return ShellStatus.CONTINUE;
			}
			path = Paths.get(list.get(0));
		} catch (Exception e) {
			env.writeln("Error occured while parsing argument line - " + arguments);
			return ShellStatus.CONTINUE;
		}
		
		path = relativize(env.getCurrentDirectory(), path);
		if(path == null ) {
			env.writeln("Provided arguments cannot be interpreted as valid path - " + path );
			return ShellStatus.CONTINUE;
		}
		
		if(Files.isDirectory(path)) {
			env.writeln("Provided path is directory" + path);
			return ShellStatus.CONTINUE;
		}
		
		try {
			printHexDump(env, path);
		} catch (Exception e) {
			env.writeln("Error while opening file -> " + path);
		}
		
		return ShellStatus.CONTINUE;
	}

	/**
	 * Prints hexadecimal and textual content on standard output
	 * @param env shell on which will be written
	 * @param path file path
	 * @throws IOException if IO exception occurs
	 */
	private void printHexDump(Environment env, Path path) throws IOException {
		try(BufferedInputStream is = new BufferedInputStream(Files.newInputStream(path))){
			int count = 0;
			while(true) {
				byte []buff = new byte[16];
				int read = is.read(buff);
				if(read == -1) break;
				String text = getText(buff,read);
				String part1 = byteToHex(Arrays.copyOfRange(buff, 0, 8)).replace("00", "");
				String part2 = byteToHex(Arrays.copyOfRange(buff, 8, 16)).replace("00", "");
				
				env.writeln(String.format("%08x: %-24s|%-24s | %s", count*16, part1, part2, text));
				++count;
			}
		}
	}
	
	/**
	 * Transforms byte array into correspodent hex values 
	 * @param b byte arrays
	 * @return hexadecimal values
	 */
	private String byteToHex(byte [] b) {
		Objects.requireNonNull(b);
		StringBuilder hex = new StringBuilder();
		
		for(byte by:b) {
			hex.append(String.format("%02x ", by));
		}
		return hex.toString().toUpperCase();
	}
	
	/**
	 * Interprests given byte array into text
	 * @param data byte array which will be interpreted 
	 * @param read number of read bytes
	 * @return interpreted byte array as text
	 */
	private String getText(byte[] data, int read) {
		StringBuilder builder = new StringBuilder();
		for(int i = 0; i < read ; ++i) {
			if(data[i] < 32 || data[i] > 127) {
				builder.append(".");
			}
			else {
				builder.append(String.valueOf((char)data[i]));
			}
		}
		return builder.toString();
	}
	
	@Override
	public String getCommandName() {
		return "hexdump";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Displays file content in hexadecimal on standard output",
				"hexdump [FILE]"
		);
	}
}
