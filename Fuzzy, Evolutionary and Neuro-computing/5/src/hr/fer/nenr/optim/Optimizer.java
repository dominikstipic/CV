package hr.fer.nenr.optim;

import hr.fer.nenr.blocks.Component;

public abstract class Optimizer {
	protected Component module;

	public Optimizer(Component module) {
		super();
		this.module = module;
	}

	public abstract void step();
	
}
