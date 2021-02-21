package hr.fer.nenr.ga.mutations;

public class AdditiveGaussianNoise extends AbstractGaussianNoise{
	
	public AdditiveGaussianNoise(double sigma, double proba) {
		super(sigma, proba);
	}

	@Override
	double specificMutation(double value) {
		return value += sample();
	}
	
}
