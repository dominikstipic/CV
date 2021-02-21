package hr.fer.nenr.test;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

import org.junit.Test;

import nenr.fuzzy.coders.DefuzzerFactory;
import nenr.fuzzy.coders.FuzzerFactory;
import nenr.fuzzy.rule.IRule;
import nenr.fuzzy.rule.Rule;
import nenr.fuzzy.utils.Utils;
import nenr.fuzzy.variables.InputVariable;
import nenr.lab1.domain.CompositeDomain;
import nenr.lab1.domain.DomainElement;
import nenr.lab1.domain.IDomain;
import nenr.lab1.domain.SimpleDomain;
import nenr.lab1.fuzzy.IFuzzySet;
import nenr.lab1.fuzzy.MutableFuzzySet;
import nenr.lab2.relations.Relations;


public class TestFuzzy {
	public static double EPSILON = 1e-3;
	
	@Test
	public void testCoa() {
		IDomain domain = SimpleDomain.intRange(6, 14);
		IFuzzySet A = Utils.fuzzyGenerator(domain,0,1./3,2./3,1,2./3,1./2,1,1./2,0);
		double d = DefuzzerFactory.COA.apply(A);
		assertEquals(10.143, d, EPSILON);
	}
	
	@Test
	public void testCardinality() {
		IDomain s1 = SimpleDomain.intRange(1, 10);
		IDomain s2 = SimpleDomain.intRange(0, 10);
		IDomain s3 = SimpleDomain.intRange(-3, 2);
		IDomain s4 = SimpleDomain.intRange(-3, 0);
		IDomain s5 = SimpleDomain.intRange(-4, -1);
		
		assertEquals(s1.getCardinality(), 10);
		assertEquals(s2.getCardinality(), 11);
		assertEquals(s3.getCardinality(), 6);
		assertEquals(s4.getCardinality(), 4);
		assertEquals(s5.getCardinality(), 4);
	}
	
	@Test
	public void test2DProduct() {
		IDomain domain = SimpleDomain.intRange(1,  4);
		IFuzzySet velik = Utils.fuzzyGenerator(domain, 0, .3, .7, 1);
		IFuzzySet mali = Utils.fuzzyGenerator(domain, 1, .7, .3, 0);
		
		MutableFuzzySet target = new MutableFuzzySet(CompositeDomain.combine(domain, domain));
		target.set(DomainElement.of(1,1), 0);
		target.set(DomainElement.of(1,2), 0);
		target.set(DomainElement.of(1,3), 0);
		target.set(DomainElement.of(1,4), 0);
		///
		target.set(DomainElement.of(2,1), .3);
		target.set(DomainElement.of(2,2), .3);
		target.set(DomainElement.of(2,3), .3);
		target.set(DomainElement.of(2,4), 0);
		///
		target.set(DomainElement.of(3,1), .7);
		target.set(DomainElement.of(3,2), .7);
		target.set(DomainElement.of(3,3), .3);
		target.set(DomainElement.of(3,4), 0);
		///
		target.set(DomainElement.of(4,1), 1);
		target.set(DomainElement.of(4,2), .7);
		target.set(DomainElement.of(4,3), .3);
		target.set(DomainElement.of(4,4), 0);
		
		IFuzzySet r1 = Relations.cartesianProduct(velik, mali);
		assertEquals(r1, target);
   }
	
	@Test
	public void test3DProduct() {
		IDomain domain = SimpleDomain.intRange(1,  4);
		IFuzzySet velik = Utils.fuzzyGenerator(domain, 0, .3, .7, 1);
		IFuzzySet srednji = Utils.fuzzyGenerator(domain, 1, .9, .5, 0);
		IFuzzySet mali = Utils.fuzzyGenerator(domain, 1, .7, .3, 0);
		
		IDomain combined = CompositeDomain.fromSimple(domain, domain, domain);
		
		MutableFuzzySet target = new MutableFuzzySet(combined);
		//1
		target.set(DomainElement.of(1,1,1), 0);
		target.set(DomainElement.of(1,1,2), 0);
		target.set(DomainElement.of(1,1,3), 0);
		target.set(DomainElement.of(1,1,4), 0);
		//
		target.set(DomainElement.of(1,2,1), .3);
		target.set(DomainElement.of(1,2,2), .3);
		target.set(DomainElement.of(1,2,3), .3);
		target.set(DomainElement.of(1,2,4), 0);
		///
		target.set(DomainElement.of(1,3,1), .7);
		target.set(DomainElement.of(1,3,2), .7);
		target.set(DomainElement.of(1,3,3), .3);
		target.set(DomainElement.of(1,3,4), 0);
		///
		target.set(DomainElement.of(1,4,1), 1);
		target.set(DomainElement.of(1,4,2), .7);
		target.set(DomainElement.of(1,4,3), .3);
		target.set(DomainElement.of(1,4,4), 0);
		
		//2
		target.set(DomainElement.of(2,1,1), 0);
		target.set(DomainElement.of(2,1,2), 0);
		target.set(DomainElement.of(2,1,3), 0);
		target.set(DomainElement.of(2,1,4), 0);
		//
		target.set(DomainElement.of(2,2,1), .3);
		target.set(DomainElement.of(2,2,2), .3);
		target.set(DomainElement.of(2,2,3), .3);
		target.set(DomainElement.of(2,2,4), 0);
		///
		target.set(DomainElement.of(2,3,1), .7);
		target.set(DomainElement.of(2,3,2), .7);
		target.set(DomainElement.of(2,3,3), .3);
		target.set(DomainElement.of(2,3,4), 0);
		///
		target.set(DomainElement.of(2,4,1), .9);
		target.set(DomainElement.of(2,4,2), .7);
		target.set(DomainElement.of(2,4,3), .3);
		target.set(DomainElement.of(2,4,4), 0);
		
		//3
		target.set(DomainElement.of(3,1,1), 0);
		target.set(DomainElement.of(3,1,2), 0);
		target.set(DomainElement.of(3,1,3), 0);
		target.set(DomainElement.of(3,1,4), 0);
		//
		target.set(DomainElement.of(3,2,1), .3);
		target.set(DomainElement.of(3,2,2), .3);
		target.set(DomainElement.of(3,2,3), .3);
		target.set(DomainElement.of(3,2,4), 0);
		///
		target.set(DomainElement.of(3,3,1), .5);
		target.set(DomainElement.of(3,3,2), .5);
		target.set(DomainElement.of(3,3,3), .3);
		target.set(DomainElement.of(3,3,4), 0);
		///
		target.set(DomainElement.of(3,4,1), .5);
		target.set(DomainElement.of(3,4,2), .5);
		target.set(DomainElement.of(3,4,3), .3);
		target.set(DomainElement.of(3,4,4), 0);
		
		//4
		target.set(DomainElement.of(4,1,1), 0);
		target.set(DomainElement.of(4,1,2), 0);
		target.set(DomainElement.of(4,1,3), 0);
		target.set(DomainElement.of(4,1,4), 0);
		//
		target.set(DomainElement.of(4,2,1), 0);
		target.set(DomainElement.of(4,2,2), 0);
		target.set(DomainElement.of(4,2,3), 0);
		target.set(DomainElement.of(4,2,4), 0);
		///
		target.set(DomainElement.of(4,3,1), 0);
		target.set(DomainElement.of(4,3,2), 0);
		target.set(DomainElement.of(4,3,3), 0);
		target.set(DomainElement.of(4,3,4), 0);
		///
		target.set(DomainElement.of(4,4,1), 0);
		target.set(DomainElement.of(4,4,2), 0);
		target.set(DomainElement.of(4,4,3), 0);
		target.set(DomainElement.of(4,4,4), 0);
		
		IFuzzySet r1 = Relations.cartesianProduct(srednji, velik, mali);
		assertEquals(r1, target);
   }
	
	@Test
	public void transformTest() {
		double x1 = Utils.transformRange(50, 0, 100, 0, 10);
		double x2 = Utils.transformRange(20, 0, 100, -5, 5);
		double x4 = Utils.transformRange(10, -90, 90, -5, 5);
		double x5 = Utils.transformRange(23, -90, 90, 0, 10);
		System.out.println("TRANSFORM: " + Arrays.asList(x1,x2,x4,x5));
		
	}
	
	@Test
	public void singletonTest() {
		InputVariable var1 = new InputVariable(20, SimpleDomain.intRange(0, 30));
		InputVariable var2 = new InputVariable(0, SimpleDomain.intRange(0, 30));
		InputVariable var3 = new InputVariable(30, SimpleDomain.intRange(0, 30));
		InputVariable var4 = new InputVariable(0, SimpleDomain.intRange(-10, 10));
		InputVariable var5 = new InputVariable(-10, SimpleDomain.intRange(-10, 10));
		InputVariable var6 = new InputVariable(10, SimpleDomain.intRange(-10, 10));
		
		IFuzzySet f1 = FuzzerFactory.SINGLETON.apply(var1);
		IFuzzySet f2 = FuzzerFactory.SINGLETON.apply(var2);
		IFuzzySet f3 = FuzzerFactory.SINGLETON.apply(var3);
		IFuzzySet f4 = FuzzerFactory.SINGLETON.apply(var4);
		IFuzzySet f5 = FuzzerFactory.SINGLETON.apply(var5);
		IFuzzySet f6 = FuzzerFactory.SINGLETON.apply(var6);
		
		System.out.println("FUZZY:");
		Utils.printFuzzy(f1, false);
		Utils.printFuzzy(f2, false);
		Utils.printFuzzy(f3, false);
		Utils.printFuzzy(f4, false);
		Utils.printFuzzy(f5, false);
		Utils.printFuzzy(f6, false);
		
	}
	
	private IFuzzySet createSingleton(int x) {
		return Utils.fuzzyGenerator(SimpleDomain.intRange(x, x), 1);
		
	}
	
	@Test
	public void inferenceTest() {
		System.out.println("----INFERENCE");
		IFuzzySet X = createSingleton(3);
		IFuzzySet Y = createSingleton(2);
		
		IDomain domain = SimpleDomain.intRange(1,  4);
		IFuzzySet velik = Utils.fuzzyGenerator(domain, 0, .3, .7, 1);
		IFuzzySet srednji = Utils.fuzzyGenerator(domain, 1, .9, .5, 0);
		IFuzzySet mali = Utils.fuzzyGenerator(domain, 1, .7, .3, 0);
		IRule rule = new Rule(List.of(velik, mali), srednji);
		BiFunction<Double, Double, Double> implication = (x, y) -> Math.min(x, y);
		
		IFuzzySet z = rule.apply(List.of(X,Y), implication);
		Utils.printFuzzy(z, false);
		double d = DefuzzerFactory.COA.apply(z);
		System.out.println(d);
		
		IFuzzySet target = Utils.fuzzyGenerator(domain, .7, .7, .5, .0);
		assertEquals(z, target);
	}
	
}
	