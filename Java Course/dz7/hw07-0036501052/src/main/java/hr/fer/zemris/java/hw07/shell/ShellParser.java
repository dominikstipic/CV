package hr.fer.zemris.java.hw07.shell;

import java.util.ArrayList;
import java.util.List;

/**
 * Parser used for text anlysis which is inputed, by user, in {@link MyShell}.
 * Analyzes command arguments,inputed by user and divides them in constituet
 * parts. It distinguish two kind of inputs. One type is normal text input and
 * another is text surrounded with qoutes where special rules are allowed.
 * 
 * @author Dominik Stipić
 *
 */
public class ShellParser {
	/**
	 * text transformed in char array
	 */
	private char data[];
	/**
	 * index of current observed char
	 */
	private int index = 0;

	/**
	 * Creates parser which will anlyze provided text
	 * 
	 * @param line
	 *            text which will be parsed
	 */
	public ShellParser(String line) {
		this.data = line.trim().toCharArray();
	}

	/**
	 * Starts processing provided text
	 * 
	 * @return list of constituent arguments
	 * @throws IllegalArgumentException - if provided text contains some 
	 * irregularities
	 */
	public List<String> process() {
		List<String> list = new ArrayList<>();
		while (index < data.length) {
			skipBlanks();
			if (data[index] == '"') {
				++index;
				if (!hasNext(index)) {
					throw new IllegalArgumentException("illegal argument format - error using \" symbol");
				}
				skipBlanks();
				if (!isPaired()) {
					throw new IllegalArgumentException("symbol \" isn't paired");
				}
				String quote = readQuotedWord();
				list.add(quote);
			} else {
				String word = readWord();
				list.add(word);
			}
		}
		return list;
	}

	/**
	 * Ignores whitespaces
	 */
	private void skipBlanks() {
		while (hasNext(index) && data[index] == ' ') {
			++index;
		}
	}

	/**
	 * Checks if the word which starts with quote, also ends with qoute. In other
	 * words checks if the quotes are paired
	 * @return true - qoutes are paired false - qoutes aren't paired
	 * 
	 */
	private boolean isPaired() {
		int i = index;
		while (hasNext(i)) {
			if (data[i] == '"' && data[i - 1] != '\\') {
				return true;
			}
			++i;
		}
		return false;
	}

	/**
	 * Reads word surrounded with qoutes
	 * @return String content inside quotes
	 * @throws IllegalArgumentException - if user inputed illegal text
	 */
	private String readQuotedWord() {
		StringBuilder builder = new StringBuilder();
		while (hasNext(index) && hasNext(index + 1)
				&& !(data[index] == '"' && Character.isWhitespace(data[index + 1]))) {
			if (data[index] == '"' && data[index - 1] != '\\') {
				throw new IllegalArgumentException("Error using \" symbol - quote symbol can't be inside word");
			}
			if (data[index] == '\\') {
				escaping(builder);
			}
			builder.append(data[index]);
			++index;
		}
		++index;
		return builder.toString();
	}

	/**
	 * Checks if escaping will occure.
	 * Symbols which can be escaped are : '"' and '\'.
	 * Symbol for escaping is : '\'
	 * @param builder buildes word from user input
	 */
	private void escaping(StringBuilder builder) {
		if (index + 1 >= data.length) {
			throw new IllegalArgumentException("illegal argument format - error using quotes");
		}
		if (data[index + 1] == '\\') {
			++index;
			return;
		} else if (data[index + 1] == '"') {
			++index;
			return;
		}
	}

	/**
	 * Reads word from text
	 * @return read word
	 * @throws IllegalArgumentException - if user inputed illegal text
	 */
	private String readWord() {
		StringBuilder builder = new StringBuilder();
		while (index < data.length && data[index] != ' ') {
			if (data[index] == '"') {
				throw new IllegalArgumentException("illegal argument format - error using \" symbol");
			}
			builder.append(data[index]);
			++index;
		}
		return builder.toString();
	}

	/**
	 * Checks if the observed array field really exists
	 * @param index of array which is observed
	 * @return true - array field isn't out of bounds
	 * false - array field is out of bounds
	 */
	private boolean hasNext(int index) {
		return index < data.length;
	}
}
