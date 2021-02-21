package hr.fer.nenr.graphic;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.util.Arrays;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;

import hr.fer.nenr.components.Listeners;
import hr.fer.nenr.components.SampleArea;
import hr.fer.nenr.models.GestureSaver;
import hr.fer.nenr.utils.Label;

public class SamplePanel extends AbstractPanel{
	private static final long serialVersionUID = 1L;
	private JRadioButton[] radioGroup;
	private JButton save;
	private JButton delete;
	private JLabel text;
	private SampleArea sampleArea;
	private GestureSaver saver;
	private int activeRadio = 0;
	public final static int ROWS = 10;
	public final static int COLS = 2; 
	public final static int LABELS_NUM = 5; 

	public SamplePanel() {
		super();
	}
	
	@Override
	protected void initComponents() {
		save = new JButton("SAVE");
	    delete = new JButton("DELETE");
	    radioGroup = new JRadioButton[LABELS_NUM];
	    sampleArea = new SampleArea(ROWS,COLS);
	    text = new JLabel("ADD GESTURES");
	    initButtons();
		
	    add(drawArea, BorderLayout.CENTER);
	    add(sampleArea, BorderLayout.EAST); 
		subscribe(drawArea);
	}
	 	
	private void initButtons() {
		radioGroup[0] = new JRadioButton(Label.ALPHA.label);
		radioGroup[1] = new JRadioButton(Label.BETA.label);
		radioGroup[2] = new JRadioButton(Label.DELTA.label);
		radioGroup[3] = new JRadioButton(Label.GAMMA.label);
		radioGroup[4] = new JRadioButton(Label.ETHA.label);
		
		JPanel buttonsPanel  = new JPanel(new GridLayout(3, 1));
		JPanel upPanel = new JPanel(new GridLayout(1,2));
		JPanel downPanel = new JPanel(new GridLayout(1, radioGroup.length));
		buttonsPanel.add(upPanel);
		buttonsPanel.add(downPanel);
		buttonsPanel.add(text);
		
		Arrays.asList(radioGroup).forEach(r -> downPanel.add(r));
		upPanel.add(save);
		upPanel.add(delete);
		add(buttonsPanel, BorderLayout.SOUTH);
		activateLetter(0);
	}

	@Override
	protected void addListeners() {
		drawArea.addMouseListener(Listeners.mouseReleased(this));
		drawArea.addMouseMotionListener(Listeners.mouseMotion(this));
		delete.addActionListener(Listeners.deleteButton(this));
		save.addActionListener(Listeners.saveSample(this));
	}
	
	private void activateLetter(int idx) {
		for(int i = 0; i < radioGroup.length; ++i) {
			boolean value;
			if(i == idx) {
				value = true;
				radioGroup[i].doClick();
			}
			else value = false;
			radioGroup[i].setEnabled(value);
		}
	}
	
	private Label currentLabel() {
		int idx = activeRadio;
		if(idx >= radioGroup.length) idx = radioGroup.length-1;
		String text = radioGroup[idx].getText();
		return Label.getLabel(text);
	}
	
	private void updateText(Label currentLabel) {
		int size;
		if(currentLabel == null) size = 0;
		else size = saver.getLabelCount(currentLabel);
		String str = String.format("LABEL %s, %s", currentLabel.label, size);
		this.text.setText(str);
	}
	
	private void disableAll() {
		save.setEnabled(false);
		delete.setEnabled(false);
		text.setText("FINISHED");
		activateLetter(-1);
		return;
	}
	
	
	/**
	 * Called when save button is clicked
	 */
	public void addGesture() {
		if(activeRadio >= radioGroup.length) {
			return;
		}
		Label label = currentLabel();
		if(saver.getLabelCount(label) == ROWS*COLS) {
			System.out.println("SAVING " + label.label);
			++activeRadio;
			activateLetter(activeRadio);
			sampleArea.clearArea();
			if(activeRadio >= radioGroup.length) disableAll();
			else updateText(currentLabel());
			repaintApp();
			return; 
		}
		if(this.currentModel.size() == 0) {
			return;
		}
		
		if(saver == null) saver = new GestureSaver();
		
		saver.save(currentModel, label);
		Dimension dim = drawArea.getSize();
		sampleArea.updateArea(currentModel, dim);
		updateText(label);
		repaintApp();
	}

}
