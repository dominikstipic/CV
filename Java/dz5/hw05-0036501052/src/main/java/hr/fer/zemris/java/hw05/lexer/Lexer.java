package hr.fer.zemris.java.hw05.lexer;

import static hr.fer.zemris.java.hw05.lexer.TokenType.AND;
import static hr.fer.zemris.java.hw05.lexer.TokenType.EOF;
import static hr.fer.zemris.java.hw05.lexer.TokenType.FIELD;
import static hr.fer.zemris.java.hw05.lexer.TokenType.OPERATOR;
import static hr.fer.zemris.java.hw05.lexer.TokenType.STRING;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Leksicki analizator koji tokenizira dani upit.
 * Moguci tipovi tokena su:STRING, OPERATOR,AND,FIELD,EOF.
 * @author Dominik Stipic
 *
 */
public class Lexer {
	/**
	 * podaci koje se treba tokenizirati
	 */
	private String data[];
	private int index;
	/**
	 * model tokena
	 */
	private Token token;
	/**
	 * Spremljeni tokeni koji su rastavljeni iz kompleksnog izaraza
	 */
	private List<Token> bufferedTokens;
	
	/**
	 * Stvara lekser koji ce analizirati dani upit
	 * @param expression
	 */
	public Lexer(String expression) {
		bufferedTokens = new ArrayList<>();
		data = expression.trim().split("\\s+(?=([^\"]*\"[^\"]*\")+[^\"]*$)");
		//Lookahead regex who matches all whitespaces that are not in quotes
		index = 0;
	}
	
	/**
	 * vraca sljedeci token
	 * @return token
	 */
	public Token next() {
		if(index == data.length && bufferedTokens.isEmpty()) {
			++index;
			return token = new Token(EOF, null);
		}
		if(index > data.length) {
			throw new LexerException("Lexer is empty");
		}
		
		if(!bufferedTokens.isEmpty()) {
			token = bufferedTokens.get(0);
			bufferedTokens.remove(0);
			return token;
		}
		
		else if(data[index].toUpperCase().equals("AND")) {
			++index;
			return token = new Token(AND,"AND");
		}
		else if(data[index].startsWith("\"") && data[index].endsWith("\"")) {
			return token = new Token(STRING,data[index++].replaceAll("\"","").trim());
		}
		else if(isOperator(data[index])) {
			return token = new Token(OPERATOR,data[index++]);
		}
		else if(isFieldName(data[index])){
			return token = new Token(FIELD,data[index++]);
		}
		
		List<Token> list = isComplexExpression(data[index]);
			
		if(list == null) {
			throw new LexerException("Unknown statement " + data[index]);
		}
		else {
			bufferedTokens.addAll(list);
			++index;
			token = bufferedTokens.get(0);
			bufferedTokens.remove(0);
			return token;
		}
	}
	
	/**
	 * provjerava dali je niz operator
	 * @param value string koje se provjerava
	 * @return istinitnosna vrijednost
	 */
	private boolean isOperator(String value) {
		switch(value.toUpperCase()) {
		case ">" : return true;
		case "<" : return true;
		case ">=" : return true;
		case "<=" : return true;
		case "=" : return true;
		case "!=" : return true;
		case "LIKE" : return true;
		default : return false;
		}
	}
	
	/**
	 * provjerava dali je niz atribut
	 * @param value string koje se provjerava
	 * @return istinitnosna vrijednost
	 */
	private boolean isFieldName(String value) {
		switch(value) {
		case "lastName" : return true;
		case "firstName" : return true;
		case "jmbag" : return true;
		default : return false;
		
		}
	}
	
	/**
	 * Provjerava dali je niz kompleksan.
	 * Kompeksan niz je tipa : xxx=,=xxx,xxx=yyy
	 * @param value izraz koji se provjerava
	 * @return razbijeni izraz u tokene
	 */
	private List<Token> isComplexExpression(String value) {
		String firstChar = Character.toString(value.charAt(0));
		String lastChar = Character.toString(value.charAt(value.length()-1));
		
		if(isOperator(lastChar)) {
			String field = value.substring(0,value.length()-1);
			if(!isFieldName(field))return null;
			return Arrays.asList(new Token(TokenType.FIELD,field),new Token(OPERATOR, lastChar));
		}
		else if (isOperator(firstChar)) {
			String str = value.substring(1,value.length());
			if(!str.matches("\".+\""))return null;
			return Arrays.asList(new Token(OPERATOR, firstChar),new Token(STRING,str.replaceAll("\"", "").trim()));
		}
		
		char[] chars = value.toCharArray();
		String operator = null;
		for(int i = 0;i < chars.length; ++i) {
			operator = String.valueOf(chars[i]);
			if(isOperator(operator)) break;
		}
		
		if(operator == null) return null;
		
		String parts[] = value.split(operator);
		if(!isFieldName(parts[0])) return null;
		if(!parts[1].matches("\".+\"")) return null;
		
		return Arrays.asList(new Token(FIELD,parts[0]),new Token(OPERATOR, operator),new Token(STRING,parts[1].replaceAll("\"", "").trim()));
	}
	
	/**
	 * Getter za token
	 * @return token
	 */
	public Token getToken() {
		return token;
	}

}
