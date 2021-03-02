package hr.fer.zemris.java.hw03.prob2;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

import org.junit.Test;

import hr.fer.zemris.java.custom.scripting.lexer.Lexer;
import hr.fer.zemris.java.custom.scripting.lexer.Token;

public class LexerTest {

	private String loader(String filename) {
		//doesnt working -> instead using loader1(beneath);
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		try (InputStream is = this.getClass().getClassLoader().getResourceAsStream(filename)) {
			byte[] buffer = new byte[1024];
			while (true) {
				int read = is.read(buffer);
				if (read < 1)
					break;
				bos.write(buffer, 0, read);
			}
			return new String(bos.toByteArray(), StandardCharsets.UTF_8);
		} catch (IOException ex) {
			System.out.println("No file");
			return null;
		}
	}

	private String loader1(String filename) {
		String text=null;
		try {
			Scanner s = new Scanner(new File(filename));
			text = "";
			while (true) {
				try {
					text += s.nextLine();
					text += "\n";
				} catch (Exception e) {
					s.close();
					e.getMessage();
					break;
				}

			}
			s.close();
			return text;
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		return text;
	}

	@Test
	public void basicTest() {
		String doc1 = loader1("src\\test\\resources/doc1.txt");
		Lexer lex = new Lexer(doc1);
		
		
		assertEquals("test ", lex.nextToken().getValue());
		assertEquals("{$", lex.nextToken().getValue());
		assertEquals("FOR", lex.nextToken().getValue());
		assertEquals("var", lex.nextToken().getValue());
		assertEquals("1", lex.nextToken().getValue());
		assertEquals("\"2 \"", lex.nextToken().getValue());

	}
	
	@Test
	public void forEscaping() {
		String doc1 = loader1("src\\test\\resources/doc2.txt");
		Lexer lex = new Lexer(doc1);
		
		assertEquals("A tag", lex.nextToken().getValue());
		assertEquals("{$", lex.nextToken().getValue());
		assertEquals("=", lex.nextToken().getValue());
		assertEquals("joe \"long \" smith", lex.nextToken().getValue());
		assertEquals("$}", lex.nextToken().getValue());
		assertEquals(".", lex.nextToken().getValue());

	}
	
	@Test
	public void forTextEscaping() {
		String doc1 = loader1("src\\test\\resources/doc3.txt");
		Lexer lex = new Lexer(doc1);
		
		assertEquals("A tag{$= \"joe \"long\" smith\"$} ", lex.nextToken().getValue());
		assertEquals("{$", lex.nextToken().getValue());
		assertEquals("FOR", lex.nextToken().getValue());
		assertEquals("i", lex.nextToken().getValue());
		assertEquals("sin", lex.nextToken().getValue());
		assertEquals("$}", lex.nextToken().getValue());

	}

}
