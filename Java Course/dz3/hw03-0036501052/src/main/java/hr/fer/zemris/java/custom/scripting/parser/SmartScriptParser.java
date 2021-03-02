package hr.fer.zemris.java.custom.scripting.parser;

import hr.fer.zemris.java.custom.scripting.elems.Element;
import hr.fer.zemris.java.custom.scripting.elems.ElementConstantDouble;
import hr.fer.zemris.java.custom.scripting.elems.ElementConstantInteger;
import hr.fer.zemris.java.custom.scripting.elems.ElementFunction;
import hr.fer.zemris.java.custom.scripting.elems.ElementOperator;
import hr.fer.zemris.java.custom.scripting.elems.ElementString;
import hr.fer.zemris.java.custom.scripting.elems.ElementVariable;
import hr.fer.zemris.java.custom.scripting.lexer.Lexer;
import hr.fer.zemris.java.custom.scripting.lexer.Token;
import hr.fer.zemris.java.custom.scripting.lexer.TokenType;
import hr.fer.zemris.java.custom.scripting.nodes.DocumentNode;
import hr.fer.zemris.java.custom.scripting.nodes.EchoNode;
import hr.fer.zemris.java.custom.scripting.nodes.ForLoopNode;
import hr.fer.zemris.java.custom.scripting.nodes.Node;
import hr.fer.zemris.java.custom.scripting.nodes.TextNode;

/**
 * Represent parser whose goal is syntax analysis of given language.
 * Reads input and creates tree structure.Document Node is top hierarchy element
 * of this tree.
 * @author Dominik Stipic
 * @version 1.0
 */
public class SmartScriptParser {
	private Lexer lexer;
	private ObjectStack stack;
	private static int FOR_LOOP_MAX_SIZE = 4;
	private DocumentNode doc;

	/**
	 * Creates parser with language which is going to be analysed
	 * @param docBody textual content of source program
	 */
	public SmartScriptParser(String docBody) {
		this.lexer = new Lexer(docBody);
		stack = new ObjectStack();
		parse();
	}

	/**
	 * Method which is doing parsing process for given source program
	 * @throws SmartScriptParserException - if the source code isnt structured properly
	 */
	private void parse() {
		doc = new DocumentNode();
		stack.push(doc);

		while (true) {
			Token token = lexer.nextToken();
			if (token.getType() == TokenType.EOF)break;
			Node node = getNode(token);
			
			if(node == null) {
				stack.pop();
				if(stack.isEmpty()) {
					throw new SmartScriptParserException("Document isn't structured properly");
				}
			}
			else if(node instanceof ForLoopNode) {
				Node popped = (Node) stack.pop();
				popped.addChildNode(node);
				stack.push(popped);
				stack.push(node);
			}
			else {
				Node popped = (Node) stack.pop();
				popped.addChildNode(node);
				stack.push(popped); 
			}
		}
		if(stack.size() != 1) {
			throw new SmartScriptParserException("Not enough END statments");
		}
	}

	public DocumentNode getDocumentNode() {
		return doc;
	}
	
	/**
	 * Gets Node which is bigger hierarchy structure than token 
	 * @param token from which decision about Node type will be made
	 * @return Node which fits given token
	 * @throws SmartScriptParserException - if the source code isnt structured properly
	 */
	private Node getNode(Token token) {
		TokenType type = token.getType();
		Node node = null;
		if (type == TokenType.WORD) {
			node = new TextNode((String) token.getValue());
		} 
		else if (type == TokenType.TAG) {
			lexer.nextToken();
			if (lexer.getToken().getType() != TokenType.KEYWORD) {
				throw new SmartScriptParserException("Keyword is missing");
			}

			if (lexer.getToken().getValue().equals(TagName.FOR.toString())) {
				node = forLoopElement();
			}
			else if (lexer.getToken().getValue().equals(TagName.ECHO.toString())) {
				node = echoElement();
			}
			else {
				//END tag name returns null
				Token test = lexer.nextToken();
				if(test.getType() != TokenType.TAG) {
					throw new SmartScriptParserException("error in END tag");
				}
				node = null;
			}
		}
		return node;
	}
	
	/**
	 * Extracts all relevant information from FOR node:
	 * start expression,end expression,variable and arbitrarily step expression
	 * @return Node- for node with its subunits
	 * @throws SmartScriptParserException - if the for loop elements are invalid or more than 4 and less then 3 
	 * expression were found
	 */
	private Node forLoopElement() {
		Token[] parametars = new Token[FOR_LOOP_MAX_SIZE];
		int index = 0;
		
		while(lexer.nextToken().getType() != TokenType.TAG) {
			if(index >= FOR_LOOP_MAX_SIZE) {
				throw new SmartScriptParserException("ForLoop must have three or four parametars");
			}
			parametars[index] = lexer.getToken();
			++index;
		}
		if(index < 3 ) {
			throw new SmartScriptParserException("ForLoop must have three or four parametars");
		}
		
		if(parametars[0].getType() != TokenType.VARIABLE) {
			throw new SmartScriptParserException("first parametar of foor loop must be variable");
		}
		
		if(index == 4) {
			if( !isNumericOrString(parametars[1], parametars[2], parametars[3]) ) {
				throw new SmartScriptParserException("start,end or step expression of for loop is invalid");
			}
		}
		else {
			if( !isNumericOrString(parametars[1], parametars[2]) ) {
				throw new SmartScriptParserException("start,end or step expression of for loop is invalid");
			}
		}
		
		Element startExpression = getElementExpression(parametars[1]);
		Element endExpession = getElementExpression(parametars[2]);
		
		if(index == 3) {
			return new ForLoopNode(new ElementVariable((String) parametars[0].getValue()) ,startExpression, endExpession, null);	
		}
		else {
			Element stepExpression = getElementExpression(parametars[3]);
			return new ForLoopNode(new ElementVariable((String) parametars[0].getValue()) ,startExpression, endExpession, stepExpression);	
		}
			
	}
	
	/**
	 * Extracts all relevant information from echo node:
	 * variables,functions,string...
	 * @return Node- echonode with its subunits
	 * @throws SmartScriptParserException - if expressions within echo node weren't 
	 * structured properly 
	 */
	private Node echoElement() {
		ObjectStack helpStack = new ObjectStack();
		
		while(lexer.getToken().getType() != TokenType.TAG) {
			Token tk = lexer.nextToken();
			Element el = getElementExpression(tk);
			helpStack.push(el);
		}
		helpStack.pop();
		if(helpStack.isEmpty()) {
			throw new SmartScriptParserException("invalid use of echo tag");
		}
		
		Element elements[] = new Element[helpStack.size()];
		
		while(!helpStack.isEmpty()) {
			elements[helpStack.size()-1] = (Element) helpStack.pop();
		}
		
		return new EchoNode(elements);
	}
	
	/**
	 * Gets element according to token type 
	 * @param token which is checked 
	 * @return element value which belongs to given token
	 */
	private Element getElementExpression(Token token) {
		if(token.getType() == TokenType.VARIABLE) {
			return new ElementVariable((String)token.getValue());
		}
		else if(token.getType() == TokenType.DECIMAL) {
			return new ElementConstantDouble((double)token.getValue());
		}
		else if(token.getType() == TokenType.STRING) {
			return new ElementString((String)token.getValue());
		}
		else if(token.getType() == TokenType.INTEGER) {
			return new ElementConstantInteger((int)token.getValue());
		}
		else if(token.getType() == TokenType.FUNCTION) {
			return new ElementFunction((String)token.getValue());
		}
		else {
			return new ElementOperator((String)token.getValue());
		}
	}
	
	/**
	 * Checks if the tokens have some numeric or string value
	 * @param tokens which are checked
	 * @return true - if the all given tokens have numeric or string value
	 * false - otherwise
	 */
	private boolean isNumericOrString(Token ...tokens) {
		for(Token token:tokens) {
			if(token.getType() != TokenType.DECIMAL && token.getType() != TokenType.INTEGER && token.getType() != TokenType.STRING) {
				return false;
			}
		}
		return true;
	}

}
