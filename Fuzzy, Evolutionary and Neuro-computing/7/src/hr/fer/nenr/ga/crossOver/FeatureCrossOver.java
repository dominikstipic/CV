package hr.fer.nenr.ga.crossOver;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;

import hr.fer.nenr.dataset.Dataset;
import hr.fer.nenr.dataset.IDataset;
import hr.fer.nenr.ga.Chromosome;
import hr.fer.nenr.ga.operators.ICrossOver;
import hr.fer.nenr.models.AssignmentProblem;
import hr.fer.nenr.models.Data;
import hr.fer.nenr.models.GeneInfo;
import hr.fer.nenr.models.NeuralNet;
import hr.fer.nenr.utils.LinAlg;
import hr.fer.nenr.utils.Utils;

public class FeatureCrossOver implements ICrossOver{
	private int[] paramSizes;
	private NeuralNet net;
	private int N = 20;
	private IDataset<Data> dataset;
	
	public FeatureCrossOver(NeuralNet net) {
		this.dataset = Dataset.build();
		this.net = net;
		List<Integer> sizes = net.paramSizeList();
		paramSizes = new int[sizes.size()];
		for(int i = 0; i < paramSizes.length; ++i) paramSizes[i] = sizes.get(i);
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
	
	private int[] findSimilarNeurons(int layerId, double[] params1, double[] params2) {
		double[][] activations1 = new double[N][net.dimensions[layerId+1]];
		double[][] activations2 = new double[N][net.dimensions[layerId+1]];
		for(int i = 0; i < N; ++i) {
			double[] input = dataset.get(Utils.randInt(0, dataset.size())).example;
			net.setParams(params1);
			double[] out1 = net.forwardLayer(layerId, input);
			net.setParams(params2);
			double[] out2 = net.forwardLayer(layerId, input);
			activations1[i] = out1;
			activations2[i] = out2;
		}
		double[][] ta1 = LinAlg.t(activations1);
		double[][] ta2 = LinAlg.t(activations2);
		double[][] similarity = LinAlg.matrixMultiply(ta1, activations2);
		Utils.printMatrix(similarity);
		for(int i = 0; i < similarity.length; ++i) {
			double norm1 = LinAlg.norm(ta1[i]);
			for(int j = 0; j < similarity.length; ++j) {
				double norm2 = LinAlg.norm(ta2[j]);
				similarity[i][j] /= (norm1*norm2);
			}
		}
		int[][] assignment = AssignmentProblem.assignmentProblem(similarity);
		int[] permuted = new int[assignment.length];
		for(int i = 0; i < permuted.length; ++i) {
			permuted[i] = LinAlg.argmax(assignment[i]);
		}
		return permuted;
	}
	
	private double[] permute(double[] source, int[] permutationMask, int[] geneLengths) {
		int num = permutationMask.length;
		double[] result = null;
		for(int i = 0; i < num; ++i) {
			double[] extracted = GeneInfo.extract(source, permutationMask[i], geneLengths).gene;
			result = ArrayUtils.addAll(result , extracted);
		}
		return result;
	}
	
	public double[] getEquivalent(double[] parent1, double[] parent2){
		double[] result = null;
		for(int layerId = 0; layerId < net.depth; ++layerId) {
			//G, S za layerid = 0
			double[] parent2Layer = getLayerParams(layerId, parent1, parent2)[1];
//			// 6,6
            int[] layerParamSizes = {paramSizes[2*layerId], paramSizes[2*layerId+1]};
			//2,2,2, 2,2,2
			int[] geneLengths = getLayerGeneLengths(layerId);
			int[] permutationMask = findSimilarNeurons(layerId, parent1, parent2);
			
			double[][] subSeq = {GeneInfo.extract(parent2Layer, 0, layerParamSizes).gene, 
		                         GeneInfo.extract(parent2Layer, 1, layerParamSizes).gene};
			int[][] sizes = {Arrays.copyOfRange(geneLengths, 0, geneLengths.length/2), 
			                 Arrays.copyOfRange(geneLengths, geneLengths.length/2, geneLengths.length)};
			
			double[] a = permute(subSeq[0], permutationMask, sizes[0]);
			double[] b = permute(subSeq[1], permutationMask, sizes[1]);
			parent2Layer = ArrayUtils.addAll(a, b);
			result = ArrayUtils.addAll(result , parent2Layer);
		}
		return result;
	}
	
	@Override
	public Chromosome crossOver(Chromosome parent1, Chromosome parent2) {
		double[] xs = parent1.solution;
		double[] ys = parent2.solution;
		if(xs.length != ys.length)
			throw new IllegalArgumentException("Vector sizes should be the same");
		ys = getEquivalent(xs, ys);
		Chromosome parent1Equivalent = new Chromosome(ys);
		ICrossOver crossOver = new NeuronCrossOver(net);
		Chromosome child = crossOver.crossOver(parent1, parent1Equivalent);
		return child;
	}

	@Override
	public String name() {
		return "FEATURE_CROSSOVER";
	}
	
}
