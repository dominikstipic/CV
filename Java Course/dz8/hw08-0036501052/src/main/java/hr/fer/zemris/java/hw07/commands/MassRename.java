package hr.fer.zemris.java.hw07.commands;

import static hr.fer.zemris.java.hw07.utils.Util.relativize;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;
import java.util.stream.Collectors;
import hr.fer.zemris.java.hw07.shell.Environment;
import hr.fer.zemris.java.hw07.shell.ShellCommand;
import hr.fer.zemris.java.hw07.shell.ShellParser;
import hr.fer.zemris.java.hw07.shell.ShellStatus;
import hr.fer.zemris.java.hw07.utils.NameBuilder;
import hr.fer.zemris.java.hw07.utils.NameBuilderInfo;
import hr.fer.zemris.java.hw07.utils.NameBuilderParser;

/**
 * Command which purpose is to rename or move large number od files.
 * Command defines 4 subcommands: filter,groups,show,execute.
 * filter - prints filtered files from directoy,
 * groups - groups labeled sequence,
 * show - renames and filters files,
 * execute - renames,filters and moves files from dest to source.
 * @author Dominik Stipić
 *
 */
public class MassRename implements ShellCommand {

	/**
	 * Shell's environment
	 */
	private Environment env;
	/**
	 * subcommands map
	 */
	private Map<String, BiConsumer<List<Path>, List<String>>> cmd;
	/**
	 * destination directory
	 */
	private Path dir2;
	{
		cmd = Map.of("filter", (e, r) -> filter(e, r), "groups", (e, r) -> groups(e, r), "show",
				(e, exp) -> show(e, exp), "execute", (e, exp) -> execute(e, exp));
	}

	@Override
	public ShellStatus executeCommand(Environment env, String arguments) {
		try {
			Objects.requireNonNull(arguments);
		} catch (NullPointerException e1) {
			env.writeln("MassRename command expects 4 or 5 arguments");
			return ShellStatus.CONTINUE;
		}
		ShellParser parser = new ShellParser(arguments);
		this.env = env;
		
		Path dir1 = null;
		Path dir2 = null;
		List<String> list;
		try {
			list = parser.process();
			if (list.size() != 4 && list.size() != 5) {
				env.writeln("MassRename command expects 4 or 5 arguments");
				return ShellStatus.CONTINUE;
			}
			dir1 = Paths.get(list.get(0));
			dir2 = Paths.get(list.get(1));
		} catch (Exception e1) {
			env.writeln("error occured while parsing input -> " + e1.getMessage());
			return ShellStatus.CONTINUE;
		}
		dir1 = relativize(env.getCurrentDirectory(), dir1);
		dir2 = relativize(env.getCurrentDirectory(), dir2);
		if(dir1 == null || dir2 == null) {
			env.writeln("Provided argument cannot be interpreted as path");
			return ShellStatus.CONTINUE;
		}
		
		if (!Files.isDirectory(dir1) || !Files.isDirectory(dir2)) {
			env.writeln("Provided path is not directory -> " + dir1 + "," + dir2);
			return ShellStatus.CONTINUE;
		}
		this.dir2 = dir2;
		
		List<String> expression = new LinkedList<>();
		expression.add(list.get(3));
		if (list.size() == 5) {
			expression.add(list.get(4));
		}
		if(cmd.containsKey(list.get(2))) {
			cmd.get(list.get(2)).accept(List.of(dir1, dir2), expression);
		}
		else {
			env.writeln("Could not find given massrename command->" + list.get(2));
		}
		return ShellStatus.CONTINUE;
	}

	/**
	 * Models filter subcommand 
	 * @param paths path to source directory
	 * @param regex filter
	 */
	private void filter(List<Path> paths, List<String> regex) {
		try {
			filteredFiles(regex.get(0), paths.get(0)).forEach(f -> env.writeln(f.getFileName().toString()));
		} catch (PatternSyntaxException e) {
			env.writeln("invalid regex -> " + regex);
			return;
		} catch (IOException e) {
			env.writeln("error while opening file -> " + paths.get(0));
		}
	}

	/**
	 * Writes captured sequnce on standard output
	 * @param paths destionation path
	 * @param regex filter
	 */
	private void groups(List<Path> paths, List<String> regex) {
		try {
			filteredFiles(regex.get(0), paths.get(0)).forEach(f -> {
				env.write(f.toString() + " ");
				Matcher m = Pattern.compile(regex.get(0)).matcher(f.toString());
				m.find();
				for (int i = 0; i <= m.groupCount(); ++i) {
					env.write(i + ": " + m.group(i) + " ");
				}
				env.writeln("");
			});
		} catch (PatternSyntaxException e) {
			env.writeln("invalid regex -> " + regex);
			return;
		} catch (IOException e) {
			env.writeln("error while opening file -> " + paths.get(0));
		}
	}

	/**
	 * Renames and filters files from given directory.
	 * @param paths source and destiantion paths
	 * @param expression expression
	 */
	private void show(List<Path> paths, List<String> expression) {
		if (expression.size() != 2) {
			env.writeln("Show command of massrename expects 5 arguments");
			return;
		}
		try {
			filteredFiles(expression.get(0), paths.get(0)).forEach(f ->{
				String newName = rename(f.getFileName().toString(), expression.get(1), expression.get(0));
				env.writeln(newName);
			});
		} catch (Exception e) {
			env.writeln(e.getMessage());
		}
	}

	/**
	 * Models execute command which filters,renames,and moves files from source to destination
	 * @param paths destination and source
	 * @param expression 
	 */
	private void execute(List<Path> paths, List<String> expression) {
		if (expression.size() != 2) {
			env.writeln("Show command of massrename expects 5 arguments");
			return;
		}
		try {
			filteredFiles(expression.get(0), paths.get(0)).forEach(f -> {
				String newName = rename(f.getFileName().toString(), expression.get(1), expression.get(0));
				try {
					Files.move(f, dir2.resolve(newName));
				} catch (IOException e) {
					env.writeln("error while moving files->" + e.getMessage());
				}
			});
		} catch (Exception e) {
			env.writeln(e.getMessage());
		}
	}

	/**
	 * Filtes file from given path and with regex
	 * @param regex filter for files
	 * @param path destination file
	 * @return List of filtererd
	 * @throws IOException - if error with reading files occurs
	 */
	private List<Path> filteredFiles(String regex, Path path) throws IOException{
		Pattern p = Pattern.compile(regex, Pattern.UNICODE_CASE | Pattern.CASE_INSENSITIVE);
		return Files.list(path)
				.filter(f -> !Files.isDirectory(f) && f.getFileName().toString().matches(p.pattern()))
				.collect(Collectors.toList());
	}
	
	/**
	 * Renames file into new one
	 * @param old file name
	 * @param expression inputed expression
	 * @param regex regex for matching
	 * @return
	 */
	private String rename(String old, String expression, String regex) {
		StringBuilder quote = new StringBuilder(expression);
		quote.append("\"");
		quote.insert(0, "\"");
		NameBuilderParser parser = new NameBuilderParser(quote.toString());
		NameBuilder builder = parser.getNameBuilder();
		Pattern p = Pattern.compile(regex, Pattern.UNICODE_CASE | Pattern.CASE_INSENSITIVE);
		Matcher matcher = p.matcher(old);
		NameBuilderInfo info = new BuilderInfo(matcher);
		builder.execute(info);
		return info.getStringBuilder().toString();
	}
	
	@Override
	public String getCommandName() {
		return "massrename";
	}

	@Override
	public List<String> getCommandDescription() {
		return List.of(
				"Makes massive renaming and moving of files from given directory to target directory.",
				"New Files can also be renamed.",
				"Provides extra commands which are:filter,groups,show and execute",
				"massrenam [DIR1][DIR2][CMD][REGEX][OPTIONAL_RENAMING]"
		);
	}
	
	/**
	 * Models {@link NameBuilderInfo}.
	 * @author Dominik Stipić
	 *
	 */
	private class BuilderInfo implements NameBuilderInfo {
		/**
		 * matches patterns
		 */
		Matcher matcher;
		/**
		 * buileds strings
		 */
		StringBuilder builder = new StringBuilder();

		/**
		 * Creates BuildeInfo
		 * @param matcher for matching patterns
		 */
		public BuilderInfo(Matcher matcher) {
			this.matcher = matcher;
		}

		@Override
		public StringBuilder getStringBuilder() {
			return builder;
		}

		@Override
		public String getGroup(int index) {
			if(!matcher.find()) {
				throw new IllegalArgumentException("cannot find any matching group");
			}
			String s = matcher.group(index);
			matcher.reset();
			return s;
		}
	}
}
