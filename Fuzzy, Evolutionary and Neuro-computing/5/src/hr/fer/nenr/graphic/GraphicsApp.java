package hr.fer.nenr.graphic;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JTabbedPane;
import javax.swing.SwingUtilities;

import hr.fer.nenr.components.FrameCurrentData;
import hr.fer.nenr.interfaces.IPublisher;
import hr.fer.nenr.interfaces.ISubscriber;

public class GraphicsApp extends JFrame implements ISubscriber, IPublisher{
	private static final long serialVersionUID = 1L;
	JTabbedPane tabbedPane = new JTabbedPane();
	AbstractPanel predictPane = new PredictPanel();
	AbstractPanel samplePane = new SamplePanel();
	private List<ISubscriber> listeners = new ArrayList<>();

	public GraphicsApp(String appName) {
		initFrame(appName);
		addListeners();
		samplePane.subscribe(this);
	}
	
	private void addListeners() {
		addComponentListener(new ComponentAdapter() {
			@Override
			public void componentResized(ComponentEvent e) {
				Dimension d = e.getComponent().getSize();
				notifySubscribers(d);
			}
		});
		
		subscribe(FrameCurrentData.getFrameCurrentData());
	}

	private void initFrame(String appName) {
		setName(appName);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setSize(700, 700);
		setLocation(new Point(300,100));
		setVisible(true);
		
		tabbedPane.add("Samples", samplePane);
        tabbedPane.add("Predict", predictPane);
        add(tabbedPane);
	}

	@Override
	public void update(Object context) {
		repaint();
	}

	@Override
	public void subscribe(ISubscriber subscriber) {
		listeners.add(subscriber);
	}

	@Override
	public void unsubscribe(ISubscriber subscriber) {
		listeners.remove(subscriber);
	}

	@Override
	public void notifySubscribers(Object object) {
		listeners.forEach(l -> l.update(object));
	}
	
	
	public static void main(String[] args) {
		String appName = "Greeks predictor"; 
		SwingUtilities.invokeLater(() -> new GraphicsApp(appName));
	}
	
}
