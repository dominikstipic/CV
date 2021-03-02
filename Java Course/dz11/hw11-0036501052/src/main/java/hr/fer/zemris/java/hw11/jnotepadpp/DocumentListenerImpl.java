package hr.fer.zemris.java.hw11.jnotepadpp;

import java.util.HashMap;
import java.util.Map;

import javax.swing.ImageIcon;
import javax.swing.JTabbedPane;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.Document;

public class DocumentListenerImpl implements DocumentListener{
	private Map<Document,Boolean> changes = new HashMap<>();
	private  JTabbedPane tabs;
	private ImageIcon red;
	
	public DocumentListenerImpl( ImageIcon red, JTabbedPane pane) {
		tabs = pane;
		this.red = red;
	}
	
	@Override
	public void removeUpdate(DocumentEvent e) {
		update(e.getDocument(), true);
	}
	
	@Override
	public void insertUpdate(DocumentEvent e) {
		update(e.getDocument(), true);
	}
	
	@Override
	public void changedUpdate(DocumentEvent e) {
		update(e.getDocument(), true);
	}
	
	public void update(Document d, boolean b) {
		tabs.setIconAt(tabs.getSelectedIndex(), red);
		changes.put(d, b);
	}
	
	public void remove(Document key) {
		changes.remove(key);
	}
	
	public Boolean getValue(Document key) {
		return changes.get(key);
	}
}
