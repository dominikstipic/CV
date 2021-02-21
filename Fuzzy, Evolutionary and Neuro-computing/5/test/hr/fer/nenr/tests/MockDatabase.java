package hr.fer.nenr.tests;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import hr.fer.nenr.models.Example;

public class MockDatabase{
	List<Example> database = new ArrayList<>();  
	
	public MockDatabase() {
		database = Arrays.asList(new Example(Arrays.asList(.05,.02), 0), 
				                 new Example(Arrays.asList(.09,.11), 0),
				                 new Example(Arrays.asList(.12, .2), 0),
				                 new Example(Arrays.asList(.15,.22), 0),
				                 new Example(Arrays.asList(.2, .25), 0),
				                 new Example(Arrays.asList(.75,.75), 1),
				                 new Example(Arrays.asList(.8, .83), 1),
				                 new Example(Arrays.asList(.82, .8), 1),
				                 new Example(Arrays.asList(.9, .89), 1),
				                 new Example(Arrays.asList(.95, .89), 1),
				                 
				                 new Example(Arrays.asList(.09,.04), 0), 
				                 new Example(Arrays.asList(.10,.10), 0),
				                 new Example(Arrays.asList(.14,.21), 0),
				                 new Example(Arrays.asList(.18,.24), 0),
				                 new Example(Arrays.asList(.22,.28), 0),
				                 new Example(Arrays.asList(.77,.78), 1),
				                 new Example(Arrays.asList(.79,.81), 1),
				                 new Example(Arrays.asList(.84,.82), 1),
				                 new Example(Arrays.asList(.94,.93), 1),
				                 new Example(Arrays.asList(.98,.99), 1));
	}
	
	public Example get(int idx) {
		return database.get(idx);
	}

	
}
