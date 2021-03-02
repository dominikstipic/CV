package hr.fer.zemris.java.tests.dataBase;

import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw05.lexer.Lexer;
import hr.fer.zemris.java.hw05.lexer.LexerException;
import hr.fer.zemris.java.hw05.lexer.Token;
import static hr.fer.zemris.java.hw05.lexer.TokenType.AND;
import static hr.fer.zemris.java.hw05.lexer.TokenType.FIELD;
import static hr.fer.zemris.java.hw05.lexer.TokenType.OPERATOR;
import static hr.fer.zemris.java.hw05.lexer.TokenType.STRING;

public class LexerTest {
	String query0 = "     firstName>\"A\"     and  lastName      LIKE     \"B*ć\""; 
	
	@Test
	public void forQuery() {
		Lexer lex = new Lexer(query0);
		
		Assert.assertEquals(new Token(FIELD,"firstName"),lex.next());
		Assert.assertEquals(new Token(OPERATOR,">"),lex.next());
		Assert.assertEquals(new Token(STRING,"A"),lex.next());
		Assert.assertEquals(new Token(AND,"AND"),lex.next());
		Assert.assertEquals(new Token(FIELD,"lastName"),lex.next());
		Assert.assertEquals(new Token(OPERATOR,"LIKE"),lex.next());
		Assert.assertEquals(new Token(STRING,"B*ć"),lex.next());
		
	}
	
	@Test (expected = LexerException.class)
	public void lexerException() {
		String query0 = "firstName>\"A\" andlastName LIKE \"B*ć\""; 
		Lexer lex = new Lexer(query0);
		lex.next();
		lex.next();
		lex.next();
		lex.next();
	}
}
