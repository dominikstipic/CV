package hr.fer.nenr.optim;

import hr.fer.nenr.blocks.Composite;

public abstract class Optimizer {
	protected Composite module;

	public Optimizer(Composite module) {
		super();
		this.module = module;
	}

	public abstract void step();
	
}
