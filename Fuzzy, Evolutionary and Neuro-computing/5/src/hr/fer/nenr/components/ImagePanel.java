package hr.fer.nenr.components;


import java.awt.Color;

import hr.fer.nenr.graphic.DrawArea;
import hr.fer.nenr.models.GestureModel;

public class ImagePanel extends DrawArea{
	private static final long serialVersionUID = 1L;
	
	public ImagePanel(GestureModel model) {
		super();
		super.stroke = 2;
		super.currentColor = Color.red;
		super.gesture = model;
	}
	
	public ImagePanel() {
		this(new GestureModel());
	}
	
	
	public boolean isFilled() {
		return super.gesture.size() != 0;
	}
	
}
