package nenr.dataset;

import java.util.List;

public interface IDataset extends Iterable<Measure>{
	Measure get(int idx);
	List<Measure> range(int i, int j);
	void store(Measure m);
	int size();
}
