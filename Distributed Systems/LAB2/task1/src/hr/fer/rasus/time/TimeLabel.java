package hr.fer.rasus.time;

import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import hr.fer.rasus.utils.Utils;

public class TimeLabel implements Comparable<TimeLabel>, Comparator<TimeLabel>{
	private String source;
	private Map<String, Integer> map = new HashMap<>();

	public TimeLabel(String source, Map<String, Integer> times) {
		this.map = times;
		this.source = source;
	}
	
	public TimeLabel() {
		// TODO Auto-generated constructor stub
	}
	
	public TimeLabel(String me, List<String> sources) {
		this.source = me;
		List<Integer> zeros = IntStream.of(new int[sources.size()]).boxed().collect(Collectors.toList());
		this.map = Utils.zip(sources, zeros);
	}
	
	public TimeLabel(String me) {
		this.source = me;
		this.map.put(me, 0);
	}
	
	
	public String getSource() {
		return source;
	}

	public Map<String, Integer> getMap() {
		return map;
	}
	
	
	public void setSource(String source) {
		this.source = source;
	}

	public void setMap(Map<String, Integer> map) {
		this.map = map;
	}

	public int myTime() {
		return map.get(source);
	}
	

	public void put(String key, int value) {
		this.map.put(key, value);
	}
	
	public int forSensor(String key) {
		return map.get(key);
	}
	
	public List<Integer> getValues(){
		return map.values().stream().collect(Collectors.toList());
	}
	
	///////////////////
	
	@Override
	public String toString() {
		String vec = map.toString();
		return vec + ", " + this.source;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((map == null) ? 0 : map.hashCode());
		result = prime * result + ((source == null) ? 0 : source.hashCode());
		return result;
	}


	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		TimeLabel other = (TimeLabel) obj;
		if (map == null) {
			if (other.map != null)
				return false;
		} else if (!map.equals(other.map))
			return false;
		if (source == null) {
			if (other.source != null)
				return false;
		} else if (!source.equals(other.source))
			return false;
		return true;
	}

	@Override
	public int compareTo(TimeLabel other) {
		return compare(this, other);
	}

	
	@Override
	public int compare(TimeLabel o1, TimeLabel o2) {
		// 1 -> this is older 
		// -1 -> other is older
		// 0 -> parallel
		Map<String, Integer> thisTime  = o1.getMap();
		Map<String, Integer> otherTime = o2.getMap();
		
		if(thisTime.size() != otherTime.size()) throw new IllegalArgumentException("Sizes doesn't match");
		if(!thisTime.keySet().equals(otherTime.keySet())) {
			
			throw new IllegalArgumentException("key sets doesn't match");
		} 
		
		Set<Integer> set = new HashSet<>();
		for(String key : thisTime.keySet()) {
			int diff = thisTime.get(key) - otherTime.get(key);
			if(diff == 0) {
				continue;
			}
			else {
				int v = diff > 0 ? 1 : -1;
				set.add(v);
			}
		}
			
		if(set.size() == 2 || set.size() == 0)  {
			return 0;
		}
		else {
			int x = set.stream().findFirst().get();
			return x;
		}
	}
	
	
}
