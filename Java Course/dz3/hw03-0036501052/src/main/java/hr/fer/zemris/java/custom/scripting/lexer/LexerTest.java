package hr.fer.zemris.java.custom.scripting.lexer;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import hr.fer.zemris.java.custom.scripting.elems.Element;
import hr.fer.zemris.java.custom.scripting.elems.ElementConstantDouble;
import hr.fer.zemris.java.custom.scripting.elems.ElementVariable;
import hr.fer.zemris.java.custom.scripting.nodes.DocumentNode;
import hr.fer.zemris.java.custom.scripting.nodes.EchoNode;
import hr.fer.zemris.java.custom.scripting.nodes.ForLoopNode;
import hr.fer.zemris.java.custom.scripting.nodes.Node;

/**
 * Demonstration of lexer use
 * @author Dominik Stipic
 *
 */
public class LexerTest {

	public static void main(String[] args) {
		String text;
		try {
			text = read("examples/doc5.txt");
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
	 * Reads textual file content from given path
	 * @param path of textual file
	 * @return String representation of file content
	 * @throws FileNotFoundException - if file not found
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

}
