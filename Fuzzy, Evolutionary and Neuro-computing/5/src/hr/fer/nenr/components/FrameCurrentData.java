package hr.fer.nenr.components;

import java.awt.Dimension;

import hr.fer.nenr.interfaces.ISubscriber;

public class FrameCurrentData implements ISubscriber{
	private Dimension currentDim;
	private static FrameCurrentData data;
	
	private FrameCurrentData() {}
	
	public static FrameCurrentData getFrameCurrentData() {
		if(data == null) {
			data = new FrameCurrentData();
		}
		return data;
	}
	
	public Dimension getDimension() {
		return currentDim;
	}
	
	@Override
	public void update(Object context) {
		currentDim = (Dimension) context;
	}
	
	
}
