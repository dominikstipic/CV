package hr.fer.zemris.java.smart_scripts_tests;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

import hr.fer.zemris.java.custom.scripting.nodes.DocumentNode;
import hr.fer.zemris.java.custom.scripting.nodes.EchoNode;
import hr.fer.zemris.java.custom.scripting.nodes.ForLoopNode;
import hr.fer.zemris.java.custom.scripting.nodes.Node;
import hr.fer.zemris.java.custom.scripting.nodes.TextNode;
import hr.fer.zemris.java.custom.scripting.parser.SmartScriptParser;

public class ParsersTest {

	public static void main(String[] args) throws IOException {
		String filepath = "examples/doc9.txt";
		String docBody = new String(Files.readAllBytes(Paths.get(filepath)), StandardCharsets.UTF_8);
		
		SmartScriptParser parser = new SmartScriptParser(docBody);
		DocumentNode document = parser.getDocumentNode();
		String originalDocumentBody = createOriginalDocumentBody(document);
		
		SmartScriptParser parser2 = new SmartScriptParser(originalDocumentBody);
		DocumentNode document2 = parser2.getDocumentNode();
		String copy = createOriginalDocumentBody(document2);
		System.out.println(originalDocumentBody);
		System.out.println(copy);
	}
	
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
