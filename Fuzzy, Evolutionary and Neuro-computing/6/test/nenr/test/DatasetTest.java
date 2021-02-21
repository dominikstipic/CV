package nenr.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import hr.fer.nenr.dataset.Dataset;
import hr.fer.nenr.dataset.Functions;
import hr.fer.nenr.dataset.MLDataset;
import hr.fer.nenr.models.Example;

public class DatasetTest {

	@Test
	public void test1() {
		MLDataset dataset = Dataset.sampleFunction(4, Functions.FUNCTION1);
		
		for(Example e : dataset) {
			System.out.println(e);
		}
		
		assertEquals(dataset.size(), 81);
		
		
	}
	
}
