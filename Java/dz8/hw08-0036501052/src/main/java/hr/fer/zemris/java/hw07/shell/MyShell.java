package hr.fer.zemris.java.hw07.shell;

import static hr.fer.zemris.java.hw07.shell.ShellStatus.CONTINUE;
import static hr.fer.zemris.java.hw07.shell.ShellStatus.TERMINATE;
import static java.util.Map.entry;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Scanner;
import java.util.SortedMap;
import java.util.TreeMap;

import hr.fer.zemris.java.hw07.commands.Cat;
import hr.fer.zemris.java.hw07.commands.Cd;
import hr.fer.zemris.java.hw07.commands.CharSets;
import hr.fer.zemris.java.hw07.commands.Copy;
import hr.fer.zemris.java.hw07.commands.CpTree;
import hr.fer.zemris.java.hw07.commands.DropD;
import hr.fer.zemris.java.hw07.commands.Help;
import hr.fer.zemris.java.hw07.commands.Hexdump;
import hr.fer.zemris.java.hw07.commands.ListD;
import hr.fer.zemris.java.hw07.commands.Ls;
import hr.fer.zemris.java.hw07.commands.MassRename;
import hr.fer.zemris.java.hw07.commands.Mkdir;
import hr.fer.zemris.java.hw07.commands.PopD;
import hr.fer.zemris.java.hw07.commands.PushD;
import hr.fer.zemris.java.hw07.commands.Pwd;
import hr.fer.zemris.java.hw07.commands.RmTree;
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
	 * Shared data
	 */
	private Map<String, Object> sharedData = new HashMap<>();
	
	/**
	 * current positioned directory
	 */
	private Path currentDirectory = Paths.get(".").toAbsolutePath().normalize();
	
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
		commands = Collections.unmodifiableSortedMap(new TreeMap<>(Map.ofEntries(
				entry("massrename", new MassRename()),
				entry("cptree", new CpTree()),
				entry("rmtree", new RmTree()),
				entry("pushd", new PushD()),
				entry("popd", new PopD()),
				entry("listd", new ListD()),
				entry("dropd", new DropD()),
				entry("cd", new Cd()),
				entry("pwd", new Pwd()),
				entry("help", new Help()),
				entry("symbol", new Symbol()),
				entry("charsets", new CharSets()),
				entry("cat", new Cat()),
				entry("ls", new Ls()),
				entry("tree", new Tree()),
				entry("copy", new Copy()),
				entry("mkdir", new Mkdir()),
				entry("hexdump", new Hexdump())
				 )));
	}
	
	@Override
	public Path getCurrentDirectory() {
		return currentDirectory;
	}
	@Override
	public void setCurrentDirectory(Path path) {
		Objects.requireNonNull(path, "Provided path must be directory");
		if(!Files.isDirectory(path) || Files.notExists(path)) {
			writeln("Provided path must be directory -> " + path);
		}
		currentDirectory = path;
	}
	@Override
	public Object getSharedData(String key) {
		Objects.requireNonNull(key, "Key can't be null");
		return sharedData.get(key);
	}	
	@Override
	public void setSharedData(String key, Object value) {
		Objects.requireNonNull(key, "Key can't be null");
		Objects.requireNonNull(value, "value can't be null");
		
		sharedData.put(key, value);
	}
	@Override
	public String readLine() throws ShellIOException {
			write(prompt + " ");
			String input = s.nextLine().trim();
			while(input.endsWith(" " + moreLiness)) {
				write(multiLine + " ");
				input = input.substring(0, input.length() - 1).trim();
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
					shell.writeln("Goodbye");
					status = TERMINATE;
					continue;
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
