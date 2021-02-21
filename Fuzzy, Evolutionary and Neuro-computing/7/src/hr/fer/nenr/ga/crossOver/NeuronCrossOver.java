package hr.fer.nenr.ga.crossOver;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.ICrossOver;
import hr.fer.nenr.models.GeneInfo;
import hr.fer.nenr.models.NeuralNet;

public class NeuronCrossOver implements ICrossOver{
	private int[] paramSizes;
	private NeuralNet net;
	
	public NeuronCrossOver(NeuralNet net) {
		this.net = net;
		List<Integer> sizes = net.paramSizeList();
		paramSizes = new int[sizes.size()];
		for(int i = 0; i < paramSizes.length; ++i) paramSizes[i] = sizes.get(i);
	}
	
	private int bernulli() {
		Random rand = new Random(System.nanoTime());
		return rand.nextInt(2);
	}
	
	private double[][] getLayerParams(int k, double[] xs, double[] ys) {
		net.setParams(xs);
		double[] r1 = net.asArray(k);
		net.setParams(ys);
		double[] r2 = net.asArray(k);
		double[][] layers = {r1,r2};
		return layers;
	}
	
	private int[] getLayerGeneLengths(int layerId) {
		List<Integer> arr = new LinkedList<>();
		for(Double[][] d : net.getLayerParams(layerId)) {
			int rows = d.length; 
			int cols = d[0].length;
			for(int i = 0; i < rows; ++i) {
				arr.add(cols);
			}
		}
		int[] lens = ArrayUtils.toPrimitive(arr.toArray(new Integer[0]));
		return lens;
	}
	
	@Override
	public Chromosome crossOver(Chromosome parent1, Chromosome parent2) {
		double[] xs = parent1.solution;
		double[] ys = parent2.solution;
		if(xs.length != ys.length)
			throw new IllegalArgumentException("Vector sizes should be the same");
		
		double[] childVec = null;
		for(int layerId = 0; layerId < net.depth; ++layerId) {
			double[][] layers = getLayerParams(layerId, xs, ys);
				
			int[] layerParamSizes = {paramSizes[2*layerId], paramSizes[2*layerId+1]};
			int neuronNum = net.dimensions[layerId+1];
			int[] geneLengths = getLayerGeneLengths(layerId);
			double[] vec1 = null; double[] vec2 = null;
			for(int neuronId = 0; neuronId < neuronNum; ++neuronId) {
				int rand = bernulli();
				double[] parent = layers[rand];
				double[][] theta = {GeneInfo.extract(parent, 0, layerParamSizes).gene, 
						            GeneInfo.extract(parent, 1, layerParamSizes).gene};
				int[][] sizes = {Arrays.copyOfRange(geneLengths, 0, geneLengths.length/2), 
						         Arrays.copyOfRange(geneLengths, geneLengths.length/2, geneLengths.length)};
				double[] gene1 = GeneInfo.extract(theta[0], neuronId, sizes[0]).gene; 
				double[] gene2 = GeneInfo.extract(theta[1], neuronId, sizes[1]).gene; 
				vec1 = ArrayUtils.addAll(vec1, gene1);
				vec2 = ArrayUtils.addAll(vec2, gene2);
			}
			double[] layerSequence = ArrayUtils.addAll(vec1, vec2);
			childVec = ArrayUtils.addAll(childVec, layerSequence);
		}
		Chromosome child = new Chromosome(childVec);
		return child;
	}

	@Override
	public String name() {
		return "NEURON_CROSSOVER";
	}
}
