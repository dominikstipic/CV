package hr.fer.nenr.graphic;

import java.awt.BorderLayout;
import java.awt.GridLayout;

import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;

import hr.fer.nenr.components.Listeners;
import hr.fer.nenr.interfaces.MLDataset;
import hr.fer.nenr.nn.NN;

public class PredictPanel extends AbstractPanel{
	private static final long serialVersionUID = 1L;
	private JButton train;
	private JButton predict;
	private JButton delete;
	private JLabel text;
	private NN net;
	private MLDataset dataset;
	public final static int LABELS_NUM = 5; 

	public PredictPanel() {
		super();
	}
	
	@Override
	protected void initComponents() {
		train = new JButton("TRAIN");
		predict = new JButton("PREDICT");
	    delete = new JButton("DELETE");
	    text = new JLabel("CLICK TRAIN TO TRAIN NEURAL NETWORK");
	    initButtons();
		
	    add(drawArea, BorderLayout.CENTER);
		subscribe(drawArea);
	}
	 	
	private void initButtons() {
		JPanel buttonsPanel  = new JPanel(new GridLayout(2, 1));
		JPanel upPanel = new JPanel(new GridLayout(1,3));
		buttonsPanel.add(upPanel);
		buttonsPanel.add(text);
		
		upPanel.add(train);
		upPanel.add(predict);
		upPanel.add(delete);
		add(buttonsPanel, BorderLayout.SOUTH);
	}

	@Override
	protected void addListeners() {
		drawArea.addMouseListener(Listeners.mouseReleased(this));
		drawArea.addMouseMotionListener(Listeners.mouseMotion(this));
		delete.addActionListener(Listeners.deleteButton(this));
		predict.addActionListener(Listeners.predicButton(this));
		train.addActionListener(Listeners.trainButton(this));
	}
	
	
	public void updateText(String str) {
		text.setText(str);
	}

	
	
	public NN getNet() {
		return net;
	}

	public void setNet(NN net) {
		this.net = net;
	}

	public MLDataset getDataset() {
		return dataset;
	}

	
	
	

	
}
