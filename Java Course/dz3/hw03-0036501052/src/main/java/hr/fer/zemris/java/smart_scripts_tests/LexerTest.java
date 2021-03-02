package hr.fer.zemris.java.smart_scripts_tests;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import hr.fer.zemris.java.custom.scripting.lexer.Lexer;
import hr.fer.zemris.java.custom.scripting.lexer.Token;
import hr.fer.zemris.java.custom.scripting.lexer.TokenType;

/**
 * Demonstration of lexer use.
 * path to txt file should be passed through command-line.
 * @author Dominik Stipic
 *
 */
public class LexerTest {

	public static void main(String[] args) {
		
		String text;
		try {
			text = read("examples/doc9.txt");
			System.out.println(text);
			System.out.println("-------------------");
			Lexer lex = new Lexer(text);
			
			while(true) {
				Token token = lex.nextToken();
				if(token.getType() == TokenType.EOF)break;
				System.out.println(token.getType() + "," + token.getValue() );
			}
			
		} catch (FileNotFoundException e) {
			System.out.println("file not found");
		}
	}
	
	/**
	 * Reads text from given textual file
	 * @param path to the textual file
	 * @return String representation of text file
	 * @throws FileNotFoundException - if the file was not found
	 */
	public static String read(String path) throws FileNotFoundException {
		Scanner s = new Scanner(new File(path));
		String text="";
		while(true) {
			try {
				text += s.nextLine();
				text +="\n";
			} catch (Exception e) {
				s.close();
				e.getMessage();
				break;
			}
			
		}
		s.close();
		return text;
	}

	String s = "domi \\\\";
}
