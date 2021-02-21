package hr.fer.nenr.components;

import java.awt.Dimension;
import java.awt.GridLayout;

import javax.swing.JPanel;

import hr.fer.nenr.models.GestureModel;
import hr.fer.nenr.utils.Preprocess;

public class SampleArea extends JPanel{
	private static final long serialVersionUID = 1L;
	private ImagePanel[][] areas;
	private int row;
	private int column;
	
	public SampleArea(int row, int column) {
		this.row = row;
		this.column = column;
		areas = new ImagePanel[row][column];
		setLayout(new GridLayout(row, column, 5, 5));
		initComponents();
	}
	
	private void initComponents() {
		for(int i = 0; i < row; ++i) {
			for(int j = 0; j < column; ++j) {
				areas[i][j] = new ImagePanel();
				areas[i][j].setPreferredSize(new Dimension(100,100));
				add(areas[i][j]);
			}
		}
	}
	
	public void clearArea() {
		for(int i = 0; i < row; ++i) {
			for(int j = 0; j < column; ++j) {
				areas[i][j].update(new GestureModel());
			}
		}
	}
	
	public void updateArea(GestureModel gesture, Dimension drawAreaDim) {
		int h = (int) areas[0][0].getPreferredSize().getHeight();
		int w = (int) areas[0][0].getPreferredSize().getWidth();
		int H = drawAreaDim.height;
		int W = drawAreaDim.width;
		gesture = Preprocess.changeInterval(gesture, H, W, h, w);
		for(int i = 0; i < row; ++i) {
			for(int j = 0; j < column; ++j) {
				boolean filled = areas[i][j].isFilled();
				if(filled == false) {
					areas[i][j].update(gesture);
					return;
				}
			}
		}
	}
	
	
	
}
