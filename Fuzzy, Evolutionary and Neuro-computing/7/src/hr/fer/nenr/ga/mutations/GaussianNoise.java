package hr.fer.nenr.ga.mutations;

public class GaussianNoise extends AbstractGaussianNoise{
	
	public GaussianNoise(double sigma, double proba) {
		super(sigma, proba);
	}

	@Override
	double specificMutation(double value) {
		return sample();
	}
	
}
