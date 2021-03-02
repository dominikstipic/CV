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

import hr.fer.zemris.java.custom.scripting.elems.Element;
import hr.fer.zemris.java.custom.scripting.elems.ElementVariable;
import hr.fer.zemris.java.custom.scripting.nodes.DocumentNode;
import hr.fer.zemris.java.custom.scripting.nodes.EchoNode;
import hr.fer.zemris.java.custom.scripting.nodes.ForLoopNode;
import hr.fer.zemris.java.custom.scripting.nodes.Node;
import hr.fer.zemris.java.custom.scripting.nodes.TextNode;
import hr.fer.zemris.java.custom.scripting.parser.SmartScriptParser;
import hr.fer.zemris.java.smart_scripts_tests.SmartScriptTester;


public class ParserTest {
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
	public void forChildren() {
			SmartScriptParser parser = new SmartScriptParser("This is sample text.{$ FOR i 1 10 1 $}");
			DocumentNode document = parser.getDocumentNode();
			Node node = (Node)document.getChild(0);
			
			assertEquals("This is sample text.", ((TextNode)node).getText());
			
			node = (Node)document.getChild(1);
			
			assertEquals("{$FOR i,1,10,1$}", ((ForLoopNode)node).toString());
	}
	
	
	@Test
	public void forChildrenSize() {
			String body = loader1("examples/doc1.txt");
			SmartScriptParser parser = new SmartScriptParser(body);
			Node document = parser.getDocumentNode();
			System.out.println(body);
			
			
			assertEquals(3, document.numberOfChildren());
			
			Node first = document.getChild(0);
			Node sec = document.getChild(1);
			Node third = document.getChild(2);
			
			assertEquals(0, first.numberOfChildren());
			assertEquals(3, sec.numberOfChildren());
			assertEquals(4, third.numberOfChildren());
		
	}
	
	@Test
	public void forEchoNode() {
		SmartScriptParser parser = new SmartScriptParser("This is {$= i @sin \"a\"  $}-th time this message is generated.");
		
		Node document = parser.getDocumentNode();
		
		assertEquals(3, document.numberOfChildren());
		
		Node first = document.getChild(0);
		Node sec = document.getChild(1);
		Node third = document.getChild(2);
		
		assertEquals("This is ", ((TextNode)first).getText());
		
		EchoNode echo = (EchoNode) sec;
		assertEquals("{$=i @sin a $}", echo.toString());
		
		assertEquals("-th time this message is generated.", ((TextNode)third).getText());
		
		
	
	}
	
}
