package hr.fer.zemris.java.hw03.prob1;

import java.util.Objects;

/**
 * Represents token created by lexical analysis.
 * Token is represeneted by two pairs : type and value
 * @author Dominik Stipic
 *
 */
public class Token {
	private TokenType type;
	private Object value;
	
	/**
	 * Constructor which creates new Token with value and type
	 * @param type of token
	 * @param value which token takes
	 */
	public Token(TokenType type, Object value) {
		this.type = Objects.requireNonNull(type);
		this.value = value;
	}

	/**
	 * getter for token type
	 * @return TokenType of this token
	 */
	public TokenType getType() {
		return type;
	}

	/**
	 * getter for token value
	 * @return token value
	 */
	public Object getValue() {
		return value;
	}
	
}

