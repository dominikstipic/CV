package hr.fer.zemris.java.hw07.utils;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.function.BooleanSupplier;
import java.util.function.Predicate;

/**
 * Parses provided expression and builds {@link NameBuilder}.
 * It has to defined states for reading expression : Groupe mode and noramal word mode.
 * @author Dominik Stipić
 */
public class NameBuilderParser {
	/**
	 * collected data
	 */
	private char[] data;
	/**
	 * index of current data
	 */
	private int index = 0;
	/**
	 * Intepreted expression
	 */
	private WholeExpression expr = new WholeExpression();
	/**
	 * checks if qoute sign is present
	 */
	private final Predicate<Character> QUOTE = (s -> s.equals('\"'));
	/**
	 * checks if whitespace sign is present
	 */
	private final Predicate<Character> WHITESPACE = (s -> s.equals(' '));

	/**
	 * checks if open symbol is encountered
	 */
	private final BooleanSupplier IS_OPEN_SYMBOL = () -> data[index] == '$' && (index + 1) < data.length && data[index + 1] == '{';
	/**
	 * checks if current symbool is the end of gruope mode
	 */
	private final BooleanSupplier FOR_GROUP = () -> data[index] != '}';;
	/**
	 * cheks if is it end of word reading
	 */
	private final BooleanSupplier FOR_WORD = () -> index < data.length  && !IS_OPEN_SYMBOL.getAsBoolean();
	/**
	 * reades qoutes
	 */
	private boolean quotes = false;
	/**
	 * currently in groupe mode
	 */
	private boolean inGroup = false;
	
	/**
	 * Creates Parser with data
	 * @param expression provided expression which need to be parsed
	 */
	public NameBuilderParser(String expression) {
		data = expression.trim().toCharArray();
	}

	/**
	 * Returns interpeted expression in {@link NameBuilder} format.
	 * @return interpreted expression
	 */
	public NameBuilder getNameBuilder() {
		Predicate<Character> tester = WHITESPACE;
		if (data[index] == '\"') {
			if(data[data.length - 1] != '"') {
				throw new IllegalArgumentException("Quotes are not paired");
			}
			tester = QUOTE;
			quotes = true;
			++index;
		}

		while (index < data.length && !tester.test(data[index])) {
			skipBlanks();
			if (IS_OPEN_SYMBOL.getAsBoolean()) {
				index += 2;
				inGroup = true;
				String group = read(FOR_GROUP).trim().replaceAll("[ ]+", "");
				inGroup = false;
				expr.nameBuilders.add(new GroupNameBuilder(group));
				++index;
			} else {
				String word = read(FOR_WORD);
				if(word.isEmpty())continue;
				expr.nameBuilders.add(new WordNameBuilder(word));
			}
		}
		
		return expr;
	}

	/**
	 * Reads the data
	 * @param tester tests if the conditon for quiting is encountered
	 * @return read data
	 */
	private String read(BooleanSupplier tester) {
		StringBuilder builder = new StringBuilder();
			try {
				while (tester.getAsBoolean()) {
					if(data[index] == '"') {
						if(!inGroup && quotes && (index + 1) == data.length)break;
						else {
							throw new IllegalArgumentException("illegal use of quotes");
						}
					}
					builder.append(data[index]);
					++index;
				}
			} catch (ArrayIndexOutOfBoundsException e) {
				throw new IllegalArgumentException("Illegal use of grouping symbols");
			}
		return builder.toString();
	}
	
	/**
	 * skips the whitespaces
	 */
	private void skipBlanks() {
		while(index < data.length && data[index] == ' ') {
			++index;
		}
	}
	

	/**
	 * NameBuilder which stores constant substring of expression
	 * @author Dominik Stipić
	 *
	 */
	private class WordNameBuilder implements NameBuilder {
		/**
		 * Stored word
		 */
		private String word;

		/**
		 * Creates WordBuilder with given word
		 * @param word read word
		 */
		public WordNameBuilder(String word) {
			this.word = word;
		}

		@Override
		public void execute(NameBuilderInfo info) {
			info.getStringBuilder().append(word);
		}
	}

	/**
	 * NameBuilder which hold interpreted data from groups.
	 * 
	 * @author Dominik Stipić
	 *
	 */
	private class GroupNameBuilder implements NameBuilder {
		/**
		 * label of observed group
		 */
		private int groupIndex;
		/**
		 * extra describtion for formating
		 */
		private int extraInfo;
		/**
		 * formating with zeroPadding?
		 */
		private boolean zeroPadding = false;
		/**
		 * Format with padding?
		 */
		private boolean padding = false;

		/**
		 * Creates NameBuilder with provided information
		 * @param symbols data 
		 */
		public GroupNameBuilder(String symbols) {
			Objects.requireNonNull(symbols, "Illegal use of grouping -> grouping index must be postive integer");
			if (!symbols.matches("[\\d]+(,[\\d]+)?")) {
				throw new IllegalArgumentException("Illegal use of grouping -> " + symbols);
			}
			if (symbols.contains(",")) {
				padding = true;
				String[] parts = symbols.split(",");
				groupIndex = Integer.valueOf(parts[0]);
				if (parts[1].startsWith("0"))zeroPadding = true;
				extraInfo = Integer.valueOf(parts[1]);
			} else {
				groupIndex = Integer.valueOf(symbols.trim());
			}
		}

		@Override
		public void execute(NameBuilderInfo info) {
			String str;
			if (padding) {
				if (zeroPadding) {
					try {
						int i = Integer.valueOf(info.getGroup(groupIndex));
						str = String.format("%0" + extraInfo + "d", i);
					} catch (NumberFormatException e) {
						throw new IllegalArgumentException("illegal use of zero padding");
					}
				}
				else {
					str = String.format("%" + extraInfo + "s", info.getGroup(groupIndex));
				}
			} else {
				str = info.getGroup(groupIndex);
			}
			info.getStringBuilder().append(str);
			}
		}

	/**
	 * Holds whole interpreted expression as list.
	 * Enables reading interpreted text by calling execute method.
	 * @author Dominik Stipić
	 *
	 */
	private class WholeExpression implements NameBuilder {
		/**
		 * Builders
		 */
		List<NameBuilder> nameBuilders = new LinkedList<>();

		@Override
		public void execute(NameBuilderInfo info) {
			nameBuilders.forEach(n -> n.execute(info));
		}
	}
}
