package hr.fer.zemris.java.hw03.prob1;

import java.util.Objects;

/**
 * Class which represents lexer machine with two possible states:
 * BASIC state - state in which lexer process 3 different data type: NUMBER,WORD,SYMBOL
 * EXTENDED state - state in which lexer process only one data type: WORD
 * text which is wanted to be processed in EXTENDED state must be surrounded by: # 
 * @author Dominik Stipic
 *@version 1.0
 */
	class Lexer {
	private char[] data;
	private Token token;
	private int currentIndex;
	private LexerState state;

	
	/**
	 * Constants used in this program
	 */
	private static final int PROCESS_TERMINATED = -1;
	private static final char WHITESPACE = ' ';
	private static final char SIGNAL_FOR_CHANGE = '#';
	private static final char ESCAPE_SIGN = '\\';

	/**
	 * Constructor which creates this lexer machine.Default state for newly created lexer is BASIC state.
	 * @param text which this lexer will tokenize.
	 */
	public Lexer(String text) {
		if (text == null) {
			throw new IllegalArgumentException("Null input is not allowed");
		}
		//removing extra-spaces, tabs, newlines, etx ...
		text = removeUnnecessarySymbols(text);
		data = text.toCharArray();

		currentIndex = 0;
		state = LexerState.BASIC;
	}

	/**
	 * Method which returns next token according to lexer state.
	 * Token consists of 2 things: type,value.
	 * Type can be NUMBER,WORD,SYMBOL 
	 * @return next Token
	 * @throws 
	 */
	public Token nextToken() {
		if (currentIndex == data.length) {
			token = new Token(TokenType.EOF, null);
			// given text was read -> process terminated
			currentIndex = PROCESS_TERMINATED; 
			return token;
		}
		if (currentIndex == PROCESS_TERMINATED) {
			throw new LexerException("There aren't more available tokens");
		}

		if (data[currentIndex] == WHITESPACE) {
			// ignore and read recursivly next Toxen;
			++currentIndex;
			token = nextToken();
		}

		else if (state == LexerState.BASIC) {
			//BASIC state funtionality 
			
			if (Character.isLetter(data[currentIndex]) || data[currentIndex] == ESCAPE_SIGN) {
				//test function ->read chars if they are letters or '\' symbol
				//action function -> check if the sign \ is applied correctly and saves correct letters.
				//note : if the \ sign is incorrectly used -> exception is thrown
				String word = read(index -> Character.isLetter(data[currentIndex]) || (data[currentIndex] == ESCAPE_SIGN) ,
						builder -> {
							if (data[currentIndex] == ESCAPE_SIGN) {
								isValid(ESCAPE_SIGN);
								++currentIndex;
							}
							builder.append(data[currentIndex]);
							++currentIndex;
						});

				token = new Token(TokenType.WORD, word);
			}

			else if (Character.isDigit(data[currentIndex])) {
				//test function ->read chars if they are digits
				//action function -> save current char representation of digit
				String str = read(index -> Character.isDigit(data[index]), builder -> {
					builder.append(data[currentIndex]);
					++currentIndex;
				});

				Long number;
				try {
					number = Long.parseLong(str);
				} catch (NumberFormatException e) {
					throw new LexerException("Invalid number type");
				}

				token = new Token(TokenType.NUMBER, number);
			}

			else {
				//Symbols 
				token = new Token(TokenType.SYMBOL, Character.valueOf(data[currentIndex]));
				++currentIndex;
			}

		}

		else {
			//EXTENDED functionality
			
			if (data[currentIndex] == SIGNAL_FOR_CHANGE) {
				// # symbol is encountered -> save as symbol
				token = new Token(TokenType.SYMBOL, Character.valueOf(data[currentIndex]));
				++currentIndex;
			}

			else {
				//all the other symbols are considered to be tokentype WORD
				//test funtion -> read chars until you encouter whitespace or #
				//action function -> save all correct chars
				String symbol = read(index -> data[currentIndex] != ' ' && data[currentIndex] != SIGNAL_FOR_CHANGE,
						builder -> {
							builder.append(data[currentIndex]);
							++currentIndex;
						});
				token = new Token(TokenType.WORD, symbol);
			}
		}

		return token;
	}

	/**
	 * Sets the lexer machine to given state
	 * @param state which lexer will obtain
	 */
	public void setState(LexerState state) {
		if (state == null) {
			throw new IllegalArgumentException("State cannot be null");
		}
		this.state = state;
		System.out.println("turning to :" + state);
	}


	/**
	 * Helper method - generic method for tokenizing inputed text
	 * @param tester object for checking specific conditions.Conditions vary according to desired behaviour.
	 * @param worker object which do the certain action according to desired behaviour.
	 * @return String represetation of obtained token
	 */
	private String read(Tester tester, Action worker) {
		StringBuilder builder = new StringBuilder();

		while (currentIndex < data.length && tester.test(currentIndex)) {
			worker.action(builder);
		}

		return builder.toString();
	}

	public Token getToken() {
		return token;
	}

	/**
	 * Helper method -> checks if the specified symbol is used properly
	 * @param symbol character which is tested from irregular use 
	 * @throws LexerException - if the symbol isn't used properly
	 */
	private void isValid(char symbol) {
		if (symbol == ESCAPE_SIGN) {
			if (currentIndex == data.length - 1) {
				throw new LexerException("Your input wasn't compilable: cause - invalid use of \\ symbol");
			}

			if (Character.isDigit(data[currentIndex + 1]) == false && data[currentIndex + 1] != '\\') {
				throw new LexerException(
						"Your input wasn't compilable: cause - '\\' symbol can be applyed just to numbers and \\");
			}
		}
	}

	/**
	 * Method which changes states of this lexer machine
	 */
	public void changeState() {
		if (state == LexerState.BASIC) {
			state = LexerState.EXTENDED;
		} else {
			state = LexerState.BASIC;
		}
	}

	/**
	 * Helper method -> method which cleans input text from unnecessary symbols
	 * @param text Which is going to be modified
	 * @return String value of modified text without unnecessary symbols
	 */
	private String removeUnnecessarySymbols(String text) {
		text = text.replace("\r", " ");
		text = text.replace("\n", " ");
		text = text.replace("\t", " ");
		text = text.trim().replaceAll("[ ]+", " ");
		return text;
	}

}
