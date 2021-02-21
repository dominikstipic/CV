package nenr.dataset;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

public class Dataset implements IDataset{
	private List<Measure> dataset = new ArrayList<Measure>();
	
	public Dataset() {}
	
	private boolean indexInRange(int i) {
		return i>= 0 && i < dataset.size();
	}
	
	@Override
	public void store(Measure m) {
		dataset.add(m);
	}

	@Override
	public Measure get(int idx) {
		if(!indexInRange(idx)) throw new IndexOutOfBoundsException("index out of dataset range");
		return dataset.get(idx);
	}

	@Override
	public List<Measure> range(int lower, int upper) {
		if(lower > upper) throw new IllegalArgumentException("upper is lower than lower");
		if(!indexInRange(lower)) throw new IndexOutOfBoundsException("index i out of dataset range");
		if(!indexInRange(upper)) throw new IndexOutOfBoundsException("index j out of dataset range");
		List<Measure> list = new ArrayList<Measure>();
		for(int i = lower; i < upper; ++i) {
			Measure m = dataset.get(i);
			list.add(m);
		}
		return list;
	}

	@Override
	public Iterator<Measure> iterator() {
		return dataset.iterator();
	}
	
	@Override
	public int size() {
		return dataset.size();
	}

	public static IDataset fromPath(String path) throws FileNotFoundException {
		File file = new File(path);
		if(!file.exists()) throw new FileNotFoundException("Cannot find specified file");
		Function<String, Double> toDouble = s -> Double.parseDouble(s);
		IDataset db = new Dataset();
		try(Scanner reader = new Scanner(file)){
			while (reader.hasNextLine()) {
		        String data = reader.nextLine();
		        String[] strs = data.split("\t");
		        double d1,d2,d3;
		        d1 = toDouble.apply(strs[0]);
		        d2 = toDouble.apply(strs[1]);
		        d3 = toDouble.apply(strs[2]);
		        Measure m = new Measure(d1,d2,d3);
		        db.store(m);
			}
		}
		return db;
		
	}
	
	
	
	
	
	

}
