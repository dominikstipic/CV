package hr.fer.zemris.java.hw05.db;

import static hr.fer.zemris.java.hw05.lexer.TokenType.AND;
import static hr.fer.zemris.java.hw05.lexer.TokenType.EOF;
import static hr.fer.zemris.java.hw05.lexer.TokenType.FIELD;
import static hr.fer.zemris.java.hw05.lexer.TokenType.OPERATOR;
import static hr.fer.zemris.java.hw05.lexer.TokenType.STRING;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Stack;

import hr.fer.zemris.java.hw05.lexer.Lexer;
import hr.fer.zemris.java.hw05.lexer.Token;

/**
 * Parsira dani upit 
 * @author Dominik Stipic
 *
 */
public class QueryParser {
	/**
	 * Leksicki analizator 
	 */
	private Lexer lex;
	/**
	 * rastavljeni upiti u dijelove
	 */
	private List<ConditionalExpression> expressions;
	/**
	 * Stog potreban za parsiranje
	 */
	private Stack<Token> stack;
	
	/**
	 * @param query upit koji se treba parsirati
	 */
	public QueryParser(String query) {
		Objects.requireNonNull(query);
		lex = new Lexer (query);
		expressions = new LinkedList<>();
		parse();
	}
	
	/**
	 * Vraca izparsirane listu izparsiranih upita sastavljenih u podupite
	 * @return lista podupita 
	 */
	public List<ConditionalExpression> getQuery(){
		return expressions;
	}
	
	/**
	 * upit je definiran kao direktan ako je oblika jmbag = "xxx"
	 * @return korespodnetna istinitnosna vrijednsot 
	 */
	public boolean isDirectQuery() {
		if(expressions.size() != 1) {
			return false;
		}
		ConditionalExpression expression = expressions.get(0);
		if(expression.getComparisonOperator() != ComparisonOperators.EQUALS) return false;
		if(expression.getFieldGetter() != FieldValueGetters.JMBAG) return false;
		return true;
	}
	
	/**
	 * Vraca jmbag u slucaju ako je upit direktan
	 * @return jmbag
	 * @throws IllegalStateException - ako upit nije direktan
	 */
	public String getQueriedJMBAG() {
		if(!isDirectQuery()) {
			throw new IllegalStateException("Operation allowed only on direct queries");
		}
		ConditionalExpression expression = expressions.get(0);
		return expression.getStringLiteral();
	}
	
	/**
	 *  Obavlja parsiranje
	 */
	private void parse() {
		stack = new Stack<>();
		while(true) {
			lex.next();
			if(lex.getToken().getType() == AND || lex.getToken().getType() == EOF) {
				if(stack.size() != 3) throw new IllegalArgumentException("Illegal query expression - AND in illegal place"); 
				Token [] arr = stack.toArray(new Token[0]);
				checkQueryCorrectness(arr);
				
				IFieldValueGetter field = getFieldValueGetter(arr[0].getValue());
				IComparisonOperator operator = getComparisonOperator(arr[1].getValue());
				expressions.add(new ConditionalExpression(field,operator,arr[2].getValue()));
				
				stack.clear();
				if(lex.getToken().getType() == EOF) break;
			}
			else {
				stack.push(lex.getToken());
			}
		}
	}
	
	/**
	 * vraca <code>FieldValueGetters</code> za zadani literal 
	 * @param field literal
	 * @return odgovarajuuci <code>FieldValueGetters</code>
	 * @throws IllegalArgumentException - ako ne postoji dani literal
	 */
	private IFieldValueGetter getFieldValueGetter(String field) {
		switch (field) {
		case "firstName":
			return FieldValueGetters.FIRST_NAME;
		case "lastName":
			return FieldValueGetters.LAST_NAME;
		case "jmbag":
			return FieldValueGetters.JMBAG;
		default:
			throw new IllegalArgumentException("error using atribute " + field);
		}
	}
	
	/**
	 * vraca <code>IComparisonOperator</code> za dani string
 	 * @param operator koji se provjerava
	 * @return IComparisonOperator za dani string 
	 */
	private IComparisonOperator getComparisonOperator(String operator) {
		switch (operator.toUpperCase()) {
		case "<":
			return ComparisonOperators.LESS;
		case ">":
			return ComparisonOperators.GREATER;
		case "<=":
			return ComparisonOperators.LESS_OR_EQUALS;
		case ">=":
			return ComparisonOperators.GREATER_OR_EQUALS;
		case "=":
			return ComparisonOperators.EQUALS;
		case "!=":
			return ComparisonOperators.NOT_EQUALS;
		case "LIKE":
			return ComparisonOperators.LIKE;
		default :
			throw new IllegalArgumentException("error using operator " + operator);
		}
	}
	
	/**
	 * provjerava dali je upit ispravno formatiran:
	 * ispravan format je tipa: atribut,operator,literal
	 * @param arr upit koji se provjerava
	 * @throws IllegalArgumentException - ako upit nije ispravno formatiran
	 */
	private void checkQueryCorrectness(Token arr[]) {
		if(arr[0].getType() != FIELD ||
		   arr[1].getType() != OPERATOR ||
		   arr[2].getType() != STRING) throw new IllegalArgumentException("incorrect query input");
	}
}
