package nenr.zad1;

import java.util.Arrays;

public class DomainElement {
	private int[] values;

	public DomainElement(int[] values) {
		this.values = values;
	}
	
	public int getNumberOfComponents() {
		return values.length;
	}

	public int getComponentValue(int n) {
		return values[n];
	}

	public int[] getValues() {
		return values;
	}

	public void setValues(int[] values) {
		this.values = values;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(values);
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
		DomainElement other = (DomainElement) obj;
		if (!Arrays.equals(values, other.values))
			return false;
		return true;
	}

	@Override
	public String toString() {
		String s = "";
		for(int value : values) {
			s += String.valueOf(value);
		}
		return s;
	}
	
//	public static DomainElement of(int[] values) {
//		return new DomainElement(values);
//	}
	
	public static DomainElement of(int value) {
		return new DomainElement(new int[] {value});
	}
	
	public static DomainElement of(int... values) {
		int[] int_values = new int[values.length];
		for(int i = 0; i < values.length; ++i) {
			int_values[i] = values[i]; 
		}
		return new DomainElement(int_values);
	}
	
}
