package hr.fer.nenr.components;

import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.List;
import java.util.stream.Collectors;

import hr.fer.nenr.graphic.AbstractPanel;
import hr.fer.nenr.graphic.PredictPanel;
import hr.fer.nenr.graphic.SamplePanel;
import hr.fer.nenr.interfaces.LossFunction;
import hr.fer.nenr.interfaces.MLDataset;
import hr.fer.nenr.loss.MSE;
import hr.fer.nenr.models.GestureModel;
import hr.fer.nenr.nn.NN;
import hr.fer.nenr.nn.NeuralDataset;
import hr.fer.nenr.optim.GradientDescent;
import hr.fer.nenr.utils.Label;
import hr.fer.nenr.utils.Preprocess;

public class Listeners {
	public static MouseMotionAdapter mouseMotion(AbstractPanel panel) {
		GestureModel model = panel.getGesture();
	    return new MouseMotionAdapter() {
			@Override
			public void mouseDragged(MouseEvent e) {
				if(panel.isDrawingLocked()) return;
				Point p = e.getPoint();
				model.add(p);
				panel.notifySubscribers(model);
			}
		}; 
	}
	
	public static MouseAdapter mouseReleased(AbstractPanel panel) {
	    return new MouseAdapter() {
			@Override
			public void mouseReleased(MouseEvent e) {
				panel.setDrawingLocked(true);
			}
		};
	}
	
	public static ActionListener deleteButton(AbstractPanel panel) {
		GestureModel model = panel.getGesture();
		return new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				panel.setDrawingLocked(false);
				model.delete();
				panel.notifySubscribers(model);
			}
		};
	}
	
	public static ActionListener saveSample(AbstractPanel panel) {
		SamplePanel samplePanel = (SamplePanel) panel;
		return new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				samplePanel.addGesture();
				panel.setDrawingLocked(false);

			}
		};
	}
	
	public static ActionListener predicButton(AbstractPanel panel) {
		PredictPanel predictPanel = (PredictPanel) panel;
		return new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				NN net = predictPanel.getNet();
				GestureModel model = predictPanel.getGesture();
				GestureModel processed = Preprocess.preprocess(model, 20);
				List<Double> input = processed.features();
				List<Double> out = net.forward(input);
				out = out.stream().map(d -> Math.round(d *100.0)/100.0).collect(Collectors.toList());
				Label label = net.argmax(out);
				String text = String.format("OUTPUT: %s, GREEK: %s", out, label.label);
				predictPanel.updateText(text);
			}
		};
	}
	
	public static ActionListener trainButton(AbstractPanel panel) {
		PredictPanel predictPanel = (PredictPanel) panel;
		int M = 20;
		MLDataset dataset = new NeuralDataset(NeuralDataset.PATH, M);
		LossFunction loss = new MSE();
		return new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				NN net = new NN(40, 4, 5);
				GradientDescent optim = new GradientDescent(net, 0.1);
				net.train(100, dataset, optim, loss, 100);
				predictPanel.setNet(net);
				predictPanel.updateText("TRAINED");
			}
		};
	}
	
}
