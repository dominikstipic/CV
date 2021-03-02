package hr.fer.zemris.java.custom.scripting.lexer;

/**
 * Enumeration which hold all possible types which lexical analysis can produce
 * @author Dominik Stipic
 *
 */
public enum TokenType {
	EOF,
	WORD,
	
	KEYWORD,
	TAG,
	OPERATOR,
	FUNCTION,
	VARIABLE,
	
	STRING,
	INTEGER,
	DECIMAL
}
