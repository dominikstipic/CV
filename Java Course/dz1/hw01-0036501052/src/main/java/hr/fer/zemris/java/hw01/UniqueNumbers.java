package hr.fer.zemris.java.hw01;

import java.util.Scanner;

/**
 * Class which implements collection of unique numbers with corresponding
 * methods for collection mangament.
 * @author Dominik Stipić
 * @version 1.0
 */
public class UniqueNumbers {
	
	/**
	 * Node which holds integer data and references on left and right TreeNode
	 * @author Dominik Stipić
	 * @version 1.0
	 */
	public static class TreeNode {
		TreeNode right;
		TreeNode left;
		int value;
	}
	
	/**
	 * Creates new TreeNode
	 * @param value Value which the node is holding 
	 * @return reference on new node
	 */
	public static TreeNode newNode(int value) {
		TreeNode node = new TreeNode ();
		node.value = value;
		return node;
		
	}
	
	/**
	 * Adds element in tree data strucuture
	 * @param root Root of tree
	 * @param value Integer data which we want add in tree
	 * @return reference on updated subtree/tree
	 */
	public static TreeNode addNode(TreeNode root, int value) {
		if(containsValue(root, value)) {
			throw new IllegalArgumentException("Argument already exist in a tree");
		}
		if(root == null) {
			return newNode(value);
		}
		else if(value > root.value ) {
			root.right = addNode(root.right,value);
		}
		else {
			root.left = addNode(root.left,value);
		}
		return root;
	} 
	
	/**
	 * Method which tells if the element is present in tree or isn't
	 * @param root Tree root
	 * @param value Positive integer 
	 * @return true = element does exist in tree
	 * 		   false = element doens't exist in tree
	 */
	public static boolean containsValue(TreeNode root,int value) {
		if(root == null) {
			return false;
		}
		if(root.value == value) {
			return true;
		}
		
		return  containsValue(root.right,value) || containsValue(root.left,value);
	}
	
	/**
	 * Counts number of element in tree data structure
	 * @param root Root of tree data structure
	 * @return The number of elements in tree
	 */
	public static int treeSize(TreeNode root) {
		if(root == null) {
			return 0;
		}
		return treeSize(root.right) + treeSize(root.left) + 1;
	}
	
	/**
	 * print elements of tree data structure 
	 * @param root Root of tree
	 * @param isAscending Label which determines print type 
	 * true = print elements in ascending manner
	 * false = print elements in descending manner
	 */
	public static void printSortedTree(TreeNode root,boolean isAscending) {
		if(root == null) {
			return;
		}
		
		if(isAscending == true) {
			printSortedTree(root.left,true);
			System.out.print(root.value + " ");
			printSortedTree(root.right,true);
		}
		else {
			printSortedTree(root.right,false);
			System.out.print(root.value + " ");
			printSortedTree(root.left,false);
		}
	}
	
	
	/**
	 * Method which is automatically called when a program starts.
	 * @param args Arguments from command-line interface
	 */
	public static void main(String[] args) {
		TreeNode tree = null;
		
		try(Scanner scanner = new Scanner(System.in)){
			while(true) {
				System.out.println("Please enter integer >");
				
				String input = scanner.next().toLowerCase();
				if(input.equals("kraj") || input.equals("end") ) { 	 //sign for stopping interaction user-program
					break;
				}
				
				try {
					int number = Integer.parseInt(input);
					tree = addNode(tree, number);
					
				} catch (NumberFormatException e) {
					System.out.println("'" + input + "' " + " isn't integer");
				} catch (IllegalArgumentException e) {
					System.out.println(e.getMessage());
				}
			}
			
			System.out.println("ascending sorted: ");
			printSortedTree(tree,true);
			
			System.out.println("\ndescending sorted: ");
			printSortedTree(tree,false);
			
			
		}
		
		
	}

}
