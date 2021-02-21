package hr.fer.nenr.blocks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import hr.fer.nenr.models.Parameters;

public class ParametrizedBlock extends Composite{

	public ParametrizedBlock(ComputationalBlock...blocks) {
		super(blocks);
	}
	
	public ParametrizedBlock(int in, int out, ComputationalBlock...blocks) {
		super(in, out, blocks);
	}
	
	public ParametrizedBlock(int in, int out) {
		super(in,out);
	}
	
	@Override
	public boolean hasParam() {
		return true;
	}

	@Override
	public void cleanGradients() {
		children.forEach(b -> {
			b.cleanGradients();
		});
	}

	@Override
	public List<Parameters> getParams() {
		List<Parameters> params = new ArrayList<>();
		for(int i = 0; i < children.size(); ++i) {
			ComputationalBlock block = children.get(i);
			if(block.hasParam()) {
				List<Parameters> param = block.getParams();
				params.addAll(param);
			}
		}
		return params;
	}

	@Override
	public void setParams(List<Parameters> params) {
		int k = 0;
		for(int i = 0; i < children.size(); ++i) {
			ComputationalBlock block = children.get(i);
			if(block.hasParam()) {
				Parameters param = params.get(k);
				block.setParams(Arrays.asList(param));
				++k;
			}
		}
	}

}
