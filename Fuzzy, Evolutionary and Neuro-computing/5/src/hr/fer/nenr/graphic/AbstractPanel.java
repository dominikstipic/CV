package hr.fer.nenr.graphic;

import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JPanel;

import hr.fer.nenr.interfaces.IPublisher;
import hr.fer.nenr.interfaces.ISubscriber;
import hr.fer.nenr.models.GestureModel;

public abstract class AbstractPanel extends JPanel implements IPublisher{
	private static final long serialVersionUID = 1L;
	private List<ISubscriber> listeners = new ArrayList<>();
	private GraphicsApp app;
	protected GestureModel currentModel = new GestureModel();
	protected DrawArea drawArea = new DrawArea();
	private boolean drawingLocked = false;

	public AbstractPanel() {
		setLayout(new BorderLayout());
		initComponents();
		addListeners();
	}
	
	public boolean isDrawingLocked() {
		return drawingLocked;
	}

	public void setDrawingLocked(boolean drawingLocked) {
		this.drawingLocked = drawingLocked;
	}

	
	protected abstract void initComponents();
	
	protected abstract void addListeners();
		
	
	@Override
	public void subscribe(ISubscriber subscriber) {
		if(subscriber.getClass().equals(GraphicsApp.class)) {
			app = (GraphicsApp) subscriber;
			return;
		}
		listeners.add(subscriber);
	}

	@Override
	public void unsubscribe(ISubscriber subscriber) {
		listeners.remove(subscriber);
	}
	
	@Override
	public void notifySubscribers(Object obj) {
		listeners.forEach(l -> l.update(obj));
	}
	
	public GestureModel getGesture() {
		return currentModel;
	}
	
	protected void repaintApp() {
		app.repaint();
		currentModel.delete();
	}
	
}
