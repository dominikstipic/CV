package hr.fer.zemris.java.hw07.shell;

import static hr.fer.zemris.java.hw07.shell.ShellStatus.CONTINUE;
import static hr.fer.zemris.java.hw07.shell.ShellStatus.TERMINATE;
import java.util.Collections;
import java.util.Map;
import java.util.Scanner;
import java.util.SortedMap;
import java.util.TreeMap;
import hr.fer.zemris.java.hw07.commands.Cat;
import hr.fer.zemris.java.hw07.commands.CharSets;
import hr.fer.zemris.java.hw07.commands.Copy;
import hr.fer.zemris.java.hw07.commands.Help;
import hr.fer.zemris.java.hw07.commands.Hexdump;
import hr.fer.zemris.java.hw07.commands.Ls;
import hr.fer.zemris.java.hw07.commands.Mkdir;
import hr.fer.zemris.java.hw07.commands.Symbol;
import hr.fer.zemris.java.hw07.commands.Tree;

/**
 * Represents an user interface for acessing operating system services.
 * It is equipped with commands which allowes: copying, listing textual file content,
 * making directories, printing directory tree, printing directory content and
 * printing binary file content as hexadecimal values.
 * @author Dominik Stipić
 *
 */
public class MyShell implements Environment,AutoCloseable{
	/**
	 * available commands
	 */
	private SortedMap<String,ShellCommand> commands;
	/**
	 * prompt symbol
	 */
	private Character prompt = '>';
	/**
	 * multiline symbol
	 */
	private Character multiLine = '|';
	/**
	 * moreline symbol
	 */
	private Character moreLiness = '\\';
	/**
	 * Scanner for reading user input
	 */
	private Scanner s = new Scanner(System.in);
	{
		commands = Collections.unmodifiableSortedMap(new TreeMap<>(Map.of(
				"help", new Help(),
				"symbol", new Symbol(),
				"charsets", new CharSets(),
				"cat", new Cat(),
				"ls", new Ls(),
				"tree", new Tree(),
				"copy", new Copy(),
				 "mkdir", new Mkdir(),
				 "hexdump", new Hexdump()
				 )));
	}
	
	@Override
	public String readLine() throws ShellIOException {
			System.out.print(prompt+" ");
			String input = s.nextLine().trim();
			String regex = "(.+)?[ ]+";
			if(moreLiness == '\\')regex += "\\";
			regex += moreLiness; 
			while(input.matches(regex)) {
				System.out.print(multiLine+" ");
				input += " " + s.nextLine().trim();
			}
			return input;
	}

	@Override
	public void write(String text) throws ShellIOException {
		System.out.print(text);
	}

	@Override
	public void writeln(String text) throws ShellIOException {
		System.out.println(text);
	}

	@Override
	public SortedMap<String, ShellCommand> commands() {
		return commands;
	}

	@Override
	public Character getMorelinesSymbol() {
		return moreLiness;
	}

	@Override
	public Character getMultilineSymbol() {
		return multiLine;
	}

	@Override
	public Character getPromptSymbol() {
		return prompt;
	}

	@Override
	public void setMultilineSymbol(Character symbol) {
		multiLine = symbol;
	}

	@Override
	public void setPromptSymbol(Character symbol) {
		prompt = symbol;
	}

	@Override
	public void setMorelinesSymbol(Character symbol) {
		moreLiness = symbol;
	}
	
	@Override
	public void close() throws Exception {
		s.close();	
	}

	/**
	 * Automatically is started when program is runned
	 * @param args unsupported
	 */
	public static void main(String[] args) {
		try(MyShell shell = new MyShell()){
			shell.writeln("Welocome to MyShell v 1.0");
			ShellStatus status = CONTINUE;
			while(status != TERMINATE) {
				String line = shell.readLine().trim();
				if(line.equals("exit")) {
					status = TERMINATE;
					continue;
				}
				if(shell.moreLiness == '\\') {
					line = line.replaceAll("([ ]+[\\\\][ ]+)", " ");
					line = line.replaceAll("([ ]+[\\\\][ ]+)", " ");
				}
				else {
					line = line.replaceAll("[ ]+"+shell.moreLiness+"[ ]+", " ");
					line = line.replaceAll("[ ]+"+shell.moreLiness+"[ ]+", " ");
				}
				String []tokens = line.split("[ ]+", 2);
				ShellCommand command = shell.commands.get(tokens[0]);
				if(command == null) {
					shell.writeln("invalid command - " + line);
					continue;
				}
				String arguments = null;
				if(tokens.length != 1 && tokens.length != 0) {
					arguments = tokens[1].trim();
				}
				status = command.executeCommand((Environment)shell, arguments);
			}
		}catch (ShellIOException e) {
			System.out.println(e.getMessage());
		}
		catch (Exception e) {
			System.out.println(e.getMessage());
		}
		
	}
}
