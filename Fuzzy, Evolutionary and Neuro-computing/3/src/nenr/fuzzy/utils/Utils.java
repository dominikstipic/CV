package nenr.fuzzy.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;
import nenr.lab1.fuzzy.IFuzzySet;
import nenr.lab1.fuzzy.MutableFuzzySet;

public class Utils {


	public static void printDomain(IDomain domain, boolean flag) {
		Consumer<String> consumer;
		consumer = flag ? s -> System.out.println(s) : s -> System.out.print(s); 
		int n = domain.getCardinality(); 
		System.out.print("{");
		for(int i = 0; i < n; ++i) {
			DomainElement el = domain.elementForIndex(i);
			consumer.accept(el.toString());
			if(i != n-1) {
				System.out.print(",");
			}
		}
		System.out.println("}");
	}
	
	public static void printFuzzy(IFuzzySet fuzzy, boolean flag) {
		Consumer<String> consumer;
		consumer = flag ? s -> System.out.println(s) : s -> System.out.print(s); 
		IDomain d = fuzzy.getDomain();
		consumer.accept("{");
		for(DomainElement el : d) {
			double val = fuzzy.getValueAt(el);
			String line = "(" + val + "/" + el + "),";
			consumer.accept(line);
		}
		System.out.println("}");
	}
	
	public static IFuzzySet fuzzyGenerator(IDomain domain, Number ...mi) {
		if(mi.length != domain.getCardinality()) {
			throw new IllegalArgumentException("Domain cardinality and a list of element weights aren't compatible");
		}
		int n = domain.getCardinality();
		MutableFuzzySet fuzzy = new MutableFuzzySet(domain);
		for(int i = 0; i < n; ++i) {
			DomainElement el = domain.elementForIndex(i);
			Number value = mi[i];
			fuzzy.set(el, value.doubleValue());
		}
		return fuzzy;
	}

	public static <T> T[] append(T[] a, T[] b) {
		List<T> arr = new ArrayList<>();
		arr.addAll(Arrays.asList(a));
		arr.addAll(Arrays.asList(b));
		T[] array = arr.toArray(a);
		return array;
	}
	
	public static List<Integer> fromArrayToList(int[] arr){
		List<Integer> list = new ArrayList<>();
		for(int x : arr) {
			list.add(x);
		}
		return list;
	}
	
	public static int[] groupIndexing(int idx, int[] arr, int[] groups) {
		int delta = groups[idx];
		int[] xs = Arrays.copyOfRange(groups, 0, idx);
		int sum = fromArrayToList(xs).stream().collect(Collectors.summingInt(Integer::intValue));
		int[] returnArr = Arrays.copyOfRange(arr, sum, sum+delta);
		return returnArr;
	}
	
	public static <T> List<T> toList(T ... xs){
		List<T> list = new ArrayList<>();
		for(T x : xs) {
			list.add(x);
		}
		return list;
	}
	
	public static double transformRange(int x, int lx, int ux, int ly, int uy) {
		int Ly = uy-ly;
		int Lx = ux-lx;
		double ratio = (double)Ly/Lx;
		double y = ly + (x-lx)*ratio;
		return y;
	}
}
