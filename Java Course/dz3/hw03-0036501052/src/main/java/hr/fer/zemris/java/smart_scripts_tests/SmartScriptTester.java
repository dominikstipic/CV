package hr.fer.zemris.java.smart_scripts_tests;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;

import hr.fer.zemris.java.custom.scripting.nodes.DocumentNode;
import hr.fer.zemris.java.custom.scripting.nodes.EchoNode;
import hr.fer.zemris.java.custom.scripting.nodes.ForLoopNode;
import hr.fer.zemris.java.custom.scripting.nodes.Node;
import hr.fer.zemris.java.custom.scripting.nodes.TextNode;
import hr.fer.zemris.java.custom.scripting.parser.SmartScriptParser;
import hr.fer.zemris.java.custom.scripting.parser.SmartScriptParserException;

/**
 * Demonstartion of SmartScriptParser.
 * path to txt file should be passed through command-line
 * @author Dominik Stipic
 *
 */
public class SmartScriptTester {

	/**
	 * Method which is automatically called when a program starts.
	 * @param args Arguments from command-line interface
	 */
	public static void main(String[] args) {
		
		//Examples -> "Examples/doc(0-6).txt" -> 6 docs
		String filepath = "examples/doc9.txt";
		String docBody = "this is {$FOR i  1 1 $}{$END$}{$FOR i 1 1 1$}";
		String docBody1;
		
		try {
			docBody1 = new String(Files.readAllBytes(Paths.get(filepath)), StandardCharsets.UTF_8);

			SmartScriptParser parser = null;
			try {
				parser = new SmartScriptParser(docBody);
			} catch (SmartScriptParserException e) {
				e.printStackTrace();
				System.out.println("Unable to parse document!");
				System.exit(-1);
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println("If this line ever executes, you have failed this class!");
				System.exit(-1);
			}
			DocumentNode document = parser.getDocumentNode();
			String originalDocumentBody = createOriginalDocumentBody(document);
			System.out.println(originalDocumentBody); // should write something like original
			// content of docBody
		} catch (IOException  | InvalidPathException e) {
			System.out.println("Filepath incorrect");
		}

	}

	/**
	 * Creates textul content from given generated parser tree 
	 * @param document-> generated parser tree
	 * @return textual content of parser tree
	 */
	public static String createOriginalDocumentBody(Node document) {
		Node node = document;
		String toReturn = "";

		for (int i = 0; i < node.numberOfChildren(); ++i) {
			Node child = node.getChild(i);
			if (child instanceof TextNode) {
				toReturn += ((TextNode) child).getText();
			} else if (child instanceof EchoNode) {
				toReturn += ((EchoNode) child).toString();
			} else {
				toReturn += ((ForLoopNode) child).toString();
				toReturn += createOriginalDocumentBody(child);
				toReturn += "{$END$}";
			}
		}

		return toReturn;
	}
}
