package hr.fer.nenr.blocks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import Jama.Matrix;
import hr.fer.nenr.models.Parameters;

public abstract class Composite implements ComputationalBlock, Iterable<ComputationalBlock>{
	protected List<ComputationalBlock> children;
	protected int in;
	protected int out;
	
	public Composite(List<ComputationalBlock> children) {
		this.children = children;
		int depth = children.size();
		this.in = children.get(0).inputFeatures();
		this.out = children.get(depth-1).outFeatures();
	}
	
	public Composite(ComputationalBlock... children) {
		this(Arrays.asList(children));
	}
	
	public Composite(int in, int out, ComputationalBlock... children) {
		this(Arrays.asList(children));
		this.in = in; this.out = out;
	}
	
	public Composite(int in, int out) {
		children = new ArrayList<>();
		this.in = in;
		this.out = out;
	}
	
	public void add(ComputationalBlock block) {
		children.add(block);
	}
	
	public void remove(ComputationalBlock block) {
		children.remove(block);
	}
	
	public void remove(int idx) {
		children.remove(idx);
	}
	
	public List<ComputationalBlock> getChildren() {
		return children;
	}
	
	@Override
	public Matrix forward(Matrix inputs) {
		for(ComputationalBlock block : children) {
			inputs = block.forward(inputs);
		}
		return inputs;
	}
	
	@Override
	public Matrix backward(Matrix gradients) {
		List<ComputationalBlock> reversed = new ArrayList<>(children);
		Collections.reverse(reversed);
		for(ComputationalBlock block : reversed) {
			gradients = block.backward(gradients);
		}
		return gradients;
	}
	
	@Override
	public int inputFeatures() {
		return in;
	}

	@Override
	public int outFeatures() {
		return out;
	}

	@Override
	public Iterator<ComputationalBlock> iterator() {
		return children.iterator();
	}

	
	//////////////// ABSTRACT /*///////////////////////////
	
	@Override
	public abstract boolean hasParam();
	
	@Override
	public abstract void cleanGradients();

	@Override
	public abstract List<Parameters> getParams();

	@Override
	public abstract void setParams(List<Parameters> params);
	
	
}
