package hr.fer.nenr.blocks;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import hr.fer.nenr.interfaces.ComputationalBlock;
import hr.fer.nenr.models.Parameters;

public abstract class Component implements ComputationalBlock, Iterable<ComputationalBlock>{
	protected List<ComputationalBlock> children;
	protected int in;
	protected int out;
	
	public Component(List<ComputationalBlock> children) {
		this.children = children;
		int depth = children.size();
		this.in = children.get(0).inputFeatures();
		this.out = children.get(depth-1).outFeatures();
	}
	
	public Component() {
		children = new ArrayList<>();
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
	public List<Double> forward(List<Double> inputs) {
		for(ComputationalBlock block : children) {
			inputs = block.forward(inputs);
		}
		return inputs;
	}
	
	@Override
	public List<Double> backward(List<Double> gradients) {
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
	public abstract Parameters getParams();

	@Override
	public abstract void setParams(Parameters params);
	
	
}
