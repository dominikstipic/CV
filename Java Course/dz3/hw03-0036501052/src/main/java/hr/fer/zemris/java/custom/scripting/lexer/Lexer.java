package hr.fer.zemris.java.custom.scripting.lexer;

import hr.fer.zemris.java.hw03.prob1.LexerException;
import hr.fer.zemris.java.hw03.prob1.Tester;

/**
 * Class which represents lexer machine with two possible states: TEXT state and
 * TAG_STATE state.TAG_STATE is executed beetween "{$" and "$}". Class will take
 * input string and tokenize it by defiend rules. Lexer machine will signal when
 * it is finshed by returning EOF token type.
 * 
 * @author Dominik Stipic
 * @version 1.0
 */
public class Lexer {
	private char[] data;
	private Token token;
	private int currentIndex;
	private LexerState state;
	private boolean textEscaping = false;
	/**
	 * Signaling that process is over
	 */
	private static final int PROCESS_TERMINATED = -1;
	/**
	 * Constant for whitespace
	 */
	private static final char WHITESPACE = ' ';
	/**
	 * Symbol which surrounds tags
	 */
	public static final String OPEN_TAG = "{$";
	public static final String CLOSE_TAG = "$}";
	/**
	 * Symbol which represents functions
	 */
	public static final char FUNCTION_ANNOTATION = '@';
	/**
	 * Symbol for denoting escape procedure
	 */
	private static final char ESCAPE_SIGN = '\\';
	/**
	 * Symbol which surrounds the string in tags
	 */
	private static final char STRING_SIGN = '"';

	/**
	 * Constructor which creates this lexer machine.Default state for newly created
	 * lexer is TEXT state.
	 * 
	 * @param text
	 *            which this lexer will tokenize.
	 */
	public Lexer(String text) {
		if (text == null) {
			throw new IllegalArgumentException("Null input is not allowed");
		}
		// removing extra-spaces etc ...
		text = text.replace("\t+", " ");
		text = text.trim().replaceAll("[ ]+", " ");
		data = text.toCharArray();

		currentIndex = 0;
		state = LexerState.TEXT;
	}

	/**
	 * Method which returns next token according to lexer state.
	 * 
	 * @return next Token
	 * @throws LexerException
	 *             - if the input text was incorrect
	 */
	public Token nextToken() {
		if (currentIndex >= data.length) {
			token = new Token(TokenType.EOF, null);
			// given text was read -> process terminated
			currentIndex = PROCESS_TERMINATED;
			return token;
		}
		if (currentIndex == PROCESS_TERMINATED) {
			throw new LexerException("There aren't more available tokens");
		}

		if (isBlankChar(currentIndex)) {
			// ignore and read recursivly next Toxen;
			++currentIndex;
			token = nextToken();
		}

		else if (isTag(currentIndex, OPEN_TAG) || isTag(currentIndex, CLOSE_TAG)) {
			if (isTag(currentIndex, OPEN_TAG)) {
				if (isTagPaired(currentIndex) == false) {
					throw new LexerException("Tags are not paired");
				}
			}
			String symbol = new String(data, currentIndex, 2);
			token = new Token(TokenType.TAG, symbol);
			currentIndex += 2;
			changeState();
		}

		else if (state == LexerState.TAG_STATE) {
			token = tagFunctionality();
		} else {
			token = textFunctionality();
		}
		return token;
	}

	/**
	 * Logic which is executed when Lexer is in TAG_STATE. Possible token types of
	 * this state are: OPERATOR,FUNCTION,VARIABLE,STRING,INTEGER,DECIMAL
	 * 
	 * @return token object
	 * @throws LexerException
	 *             - if the input text was incorrect
	 */
	private Token tagFunctionality() {
		if (Character.isLetter(data[currentIndex])) {
			String text = read(index -> !isBlankChar(currentIndex) && !isTag(currentIndex, CLOSE_TAG), word -> {
			}, // has no additonal action
					() -> {
						if (!Character.isLetter(data[currentIndex]) && !Character.isDigit(data[currentIndex])
								&& data[currentIndex] != '_') {
							return false;
						}
						return true;
					});

			if (text.toUpperCase().equals("END")) {
				token = new Token(TokenType.KEYWORD, "END");
			} else if (text.toUpperCase().equals("FOR")) {
				token = new Token(TokenType.KEYWORD, "FOR");
			} else {
				token = new Token(TokenType.VARIABLE, text);
			}
		}

		else if (isOperation(data[currentIndex])) {
			token = new Token(TokenType.OPERATOR, String.valueOf(data[currentIndex]));
			++currentIndex;
		}

		else if (data[currentIndex] == STRING_SIGN
				|| (data[currentIndex] == ESCAPE_SIGN && data[currentIndex + 1] == STRING_SIGN)) {
			String str = readString();
			token = new Token(TokenType.STRING, str);
		}

		else if (Character.isDigit(data[currentIndex]) || data[currentIndex] == '-') {
			// will throw if error
			String number = readDigit(currentIndex);
			currentIndex += number.length();

			if (number.contains(".")) {
				token = new Token(TokenType.DECIMAL, Double.parseDouble(number));
			} else {
				token = new Token(TokenType.INTEGER, Integer.parseInt(number));
			}
		} else if (data[currentIndex] == FUNCTION_ANNOTATION) {
			if (Character.isLetter(data[currentIndex + 1]) == false) {
				throw new LexerException("Error using functions");
			}
			++currentIndex;
			String funtion = read(index -> !isBlankChar(index) && !isTag(currentIndex, CLOSE_TAG), word -> {
			}, // has no additonal action
					() -> {
						if (!Character.isLetter(data[currentIndex]) && !Character.isDigit(data[currentIndex])
								&& data[currentIndex] != '_') {
							return false;
						}
						return true;
					});

			token = new Token(TokenType.FUNCTION, funtion);
		} else if (data[currentIndex] == '=') {
			token = new Token(TokenType.KEYWORD, "=");
			++currentIndex;
		} else {
			throw new LexerException("Unknown sign in tagMode - " + data[currentIndex] );
		}

		return token;
	}

	/**
	 * Logic which is executed during TEXT state of lexer machine. Lexer in this
	 * state can read only WORD token types
	 * 
	 * @return
	 */
	private Token textFunctionality() {
		Token type = null;
		String text = read(index -> !isTag(currentIndex, OPEN_TAG), word -> {
			if (data[currentIndex] == '\\' && isTag(currentIndex + 1, OPEN_TAG)) {
				textEscaping = true;
				word += OPEN_TAG;
				++currentIndex;
			} else if (data[currentIndex] == '\\' && hasNext(currentIndex + 1) && data[currentIndex + 1] == '\\' && !textEscaping) {
				word += "\\";
				++currentIndex;
			} else if (data[currentIndex] == '\\' && !textEscaping) {
				throw new LexerException("Illegal use of \\ in text mode");
			}

		}, () -> true); // has no additonal conditon

		type = new Token(TokenType.WORD, text);
		return type;
	}

	/**
	 * Generic method for tokenizing inputed text.
	 * 
	 * @param tester
	 *            Condition which secures iteration of while loop.
	 * @param cond
	 *            Specific conditon,when occured will throw LexerException
	 * @param action
	 *            object which do the certain action according to desired behaviour.
	 * @return String represetation of obtained token
	 */
	private String read(Tester tester, Action action, Condition cond) {
		String word = "";

		while (currentIndex < data.length && tester.test(currentIndex) ) {
			if (cond.condition() == false) {
				throw new LexerException("Error while lexing -> undefined sign :" + data[currentIndex]);
			}
			action.action(word);
			word += data[currentIndex];
			++currentIndex;
		}
		textEscaping = false;
		
		return word;
	}

	/**
	 * Method intended for reading digits in TAG_STATE
	 * 
	 * @param index
	 * @return String representation of read number
	 * @throws LexerException
	 *             - when number wasn't decimal or integer
	 */
	private String readDigit(int index) {
		int i = currentIndex;
		String number = "";
		while (!isBlankChar(i) && data[i] != '\n' && data[i] != '\r') {
			if (isTag(i, CLOSE_TAG))
				break;
			number += data[i];
			++i;
		}
		if (number.matches("[-]?[0-9]+([.][0-9]+)?")) {
			return number;
		} else {
			throw new LexerException("Error while tokenizing number:" + number + ",");
		}
	}

	/**
	 * Method intended for reading strings in TAG_STATE
	 * 
	 * @return String which is read in TAG_STATE
	 * @throws LexerException
	 *             - if the String isn't formated correctly.
	 */
	private String readString() {
		String str = "";
		int i = currentIndex + 1;
		// checking if the String_sign is paired
		boolean isString = false;
		while (!isTag(i, CLOSE_TAG)) {
			if (data[i] == STRING_SIGN && data[i - 1] != ESCAPE_SIGN) {
				isString = true;
				break;
			}
			++i;
		}

		if (isString) {
			str = new String(data, currentIndex + 1, i - currentIndex - 1);
			// replacing string content with appropriate symbols
			str = transformString(str);
			currentIndex = i + 1;
			return str;
		} else {
			throw new LexerException("Error using strings -> String " + str);
		}
	}

	/**
	 * Method which changes state of this lexer machine
	 */
	public void changeState() {
		if (state == LexerState.TEXT) {
			state = LexerState.TAG_STATE;
		} else {
			state = LexerState.TEXT;
		}
	}

	/**
	 * Checks if the Tag note is encountered
	 * 
	 * @param index
	 *            - beginning of tag
	 * @param tagType
	 *            - Supposed to be "{$" or "$}"
	 * @return appropriate boolean value
	 */
	private boolean isTag(int index, String tagType) {
		if (hasNext(index)) {
			String symbol = new String(data, index, 2);
			return symbol.equals(tagType) ? true : false;
		}
		return false;
	}

	/**
	 * Checks if the lexer has next token
	 * 
	 * @param index
	 *            - of checked char element
	 * @return appropriate boolean value
	 */
	private boolean hasNext(int index) {
		return index >= data.length - 1 ? false : true;
	}

	/**
	 * returns last read token
	 * 
	 * @return current available token
	 */
	public Token getToken() {
		return token;
	}

	/**
	 * Checks if the OPEN_TAG symbol "{$" has its pair "$}"
	 * 
	 * @param index
	 *            of OPEN_TAG symbol "{$"
	 * @return true - if tags are paired. false - otherwise
	 */
	private boolean isTagPaired(int index) {
		while (hasNext(index)) {
			if (isTag(index, CLOSE_TAG)) {
				return true;
			}
			++index;
		}
		return false;
	}

	/**
	 * Checks if the input char is blank char
	 * 
	 * @param index
	 *            of element which is checked
	 * @return appropriate boolean value
	 */
	private boolean isBlankChar(int index) {
		if (data[index] == WHITESPACE || data[index] == '\n' || data[index] == '\r') {
			return true;
		}
		return false;
	}

	/**
	 * Checks if the input char is one of the following mathematical operations: +,
	 * -, *, /, ^.
	 * 
	 * @param c
	 *            char which is checked
	 * @return appropriate boolean value
	 */
	private boolean isOperation(char c) {
		if (c == '-' && Character.isDigit(data[currentIndex + 1])) {
			return false;
		}
		if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
			return true;
		}
		return false;

	}

	/**
	 * Uses regexes for transforming string to appropriate form:<pre>\\ sequence treat as
	 * a single string character</pre> 
	 * <pre>\ \" treat as a single string character " (and not the end of the string)</pre>
	 * <pre>\n, \r and \t have its usual meaning </pre> 
	 * 
	 * @param str
	 *            which is going to be transformed using defined operation
	 * @return transormed string
	 */
	private String transformString(String str) {
		str = str.replaceAll("[ ]+", " ");
		str = str.replaceAll("\\\\{2}", "\\\\");
		str = str.replaceAll("\\\\n", "\n");
		str = str.replaceAll("\\\\r", "\r");
		str = str.replaceAll("\\\\t", "\t");
		str = str.replaceAll("\\\\\"", "\"");
		return str;
	}
}
