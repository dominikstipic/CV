package nenr.test;

import static org.junit.Assert.assertNotEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import hr.fer.nenr.GAConfigurator;
import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.crossOver.FeatureCrossOver;
import hr.fer.nenr.ga.crossOver.NeuronCrossOver;
import hr.fer.nenr.ga.mutations.AdditiveGaussianNoise;
import hr.fer.nenr.ga.mutations.GaussianNoise;
import hr.fer.nenr.ga.mutations.MutationComposition;
import hr.fer.nenr.ga.operators.ICrossOver;
import hr.fer.nenr.ga.operators.IMutation;
import hr.fer.nenr.models.GeneInfo;
import hr.fer.nenr.models.NeuralNet;
import hr.fer.nenr.utils.LinAlg;

public class GaTest {
	
	public static IMutation MUTATION = new MutationComposition(ls(new AdditiveGaussianNoise(1, 0.99),
														          new AdditiveGaussianNoise(5, 0.99),
														          new GaussianNoise(1, 0.6)),
														          1,1,1);
	
	public static List<IMutation> ls(IMutation ...muts) {
		List<IMutation> list = new ArrayList<>();
		for(IMutation m : muts) list.add(m);
		return list;
	}

	public void mutationTest() {
		Chromosome chr = new Chromosome(3);
		IMutation mutation = GAConfigurator.MUTATION;
		Chromosome mutated = mutation.mutate(chr);
		assertNotEquals(mutated, chr);
	}
	
	
	private NeuralNet getNet() {
		NeuralNet net = new NeuralNet(2,3,4,3);
		double[] L11 = LinAlg.rep(0., 2*3);
		double[] L12 = LinAlg.rep(1., 2*3);
		double[] L2 = LinAlg.rep(1., 12+4);
		double[] L3 = LinAlg.rep(2., 12+3);
		double[] params = org.apache.commons.lang3.ArrayUtils.addAll(L11,L12);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L2);
		params = org.apache.commons.lang3.ArrayUtils.addAll(params, L3);
		net.setParams(params);
		return net;
	}
	
	public void geneExtraction() {
		double[] genome = {1,2,  3,4,5,  6,7,8,9,  10,11,12,13,  14,15};
		int[] lens = {2,3,4,4,2};
		GeneInfo info1 = GeneInfo.extract(genome, 0, lens);
		GeneInfo info2 = GeneInfo.extract(genome, 1, lens);
		GeneInfo info3 = GeneInfo.extract(genome, 2, lens);
		GeneInfo info4 = GeneInfo.extract(genome, 3, lens);
		GeneInfo info5 = GeneInfo.extract(genome, 4, lens);
		
		System.out.println(info1);
		System.out.println(info2);
		System.out.println(info3);
		System.out.println(info4);
		System.out.println(info5);
		
	}
	
	public void neuronCrossOver() {
		NeuralNet net1 = getNet();
		double[] params1 = LinAlg.rep(0.0, net1.paramSize);
		double[] params2 = LinAlg.rep(1.0, net1.paramSize);
		
		Chromosome c1 = new Chromosome(params1);
		Chromosome c2 = new Chromosome(params2);
		ICrossOver crossOver = new NeuronCrossOver(net1);
		Chromosome child = crossOver.crossOver(c1, c2);
		System.out.println(child);
	}
	
	@Test
	public void featureCrossOver() {
		NeuralNet net1 = getNet();
		double[] params1 = LinAlg.arange(net1.paramSize);
		params1 = LinAlg.minus(params1, LinAlg.rep(2.0, params1.length));
		double[] params2 = LinAlg.arange(net1.paramSize);
		
		FeatureCrossOver crossOver = new FeatureCrossOver(net1);
		Chromosome child = crossOver.crossOver(new Chromosome(params1), new Chromosome(params2));
		System.out.println(child);
	}
}
