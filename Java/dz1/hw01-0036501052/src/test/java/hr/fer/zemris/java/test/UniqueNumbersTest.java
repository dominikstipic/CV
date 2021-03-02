package hr.fer.zemris.java.test;




import org.junit.Assert;
import org.junit.Test;

import hr.fer.zemris.java.hw01.UniqueNumbers;
import hr.fer.zemris.java.hw01.UniqueNumbers.TreeNode;

public class UniqueNumbersTest {
	
	@Test (expected = IllegalArgumentException.class)
	public void forAddingSameElement() {
		TreeNode root = null; 
		
		root = UniqueNumbers.addNode(root, 1);
		root = UniqueNumbers.addNode(root, 2);
		root = UniqueNumbers.addNode(root, 3);
		root = UniqueNumbers.addNode(root, 1);
		root = UniqueNumbers.addNode(root, 3);
		root = UniqueNumbers.addNode(root, 2);
	}
	
	@Test
	public void forTreeSize() {
		TreeNode root = null; 
		Assert.assertEquals(0, UniqueNumbers.treeSize(root));
		
		root = UniqueNumbers.addNode(root, 1);
		root = UniqueNumbers.addNode(root, 2);
		root = UniqueNumbers.addNode(root, 3);
		root = UniqueNumbers.addNode(root, 4);
		root = UniqueNumbers.addNode(root, -5);
		root = UniqueNumbers.addNode(root, 0);
		
		Assert.assertEquals(6, UniqueNumbers.treeSize(root));
	}
	
	@Test
	public void forContainingValue() {
		TreeNode root = null; 
		
		root = UniqueNumbers.addNode(root, 1);
		root = UniqueNumbers.addNode(root, 2);
		root = UniqueNumbers.addNode(root, 3);
		root = UniqueNumbers.addNode(root, 4);
		root = UniqueNumbers.addNode(root, -5);
		root = UniqueNumbers.addNode(root, 0);
		
		Assert.assertEquals(true,UniqueNumbers.containsValue(root, 1));
		Assert.assertEquals(true,UniqueNumbers.containsValue(root, 2));
		Assert.assertNotEquals(true, UniqueNumbers.containsValue(root, 10));
		Assert.assertEquals(false,UniqueNumbers.containsValue(root, 10));
	}
}
