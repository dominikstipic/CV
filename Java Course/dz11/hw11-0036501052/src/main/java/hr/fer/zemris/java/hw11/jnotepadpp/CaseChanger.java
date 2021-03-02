package hr.fer.zemris.java.hw11.jnotepadpp;

import java.awt.event.ActionEvent;
import java.util.function.Supplier;
import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;

public class CaseChanger extends JMenu{
	private static final long serialVersionUID = -5819813138446357565L;
	
	private Action action = new AbstractAction() {
		private static final long serialVersionUID = -4856968934381707305L;
		@Override
		public void actionPerformed(ActionEvent e) {
			Document doc = getTextArea.get().getDocument();
			int len = Math.abs(
					getTextArea.get().getCaret().getDot()-getTextArea.get().getCaret().getMark());
			int offset = 0;
			if(len != 0) {
				offset = Math.min(getTextArea.get().getCaret().getDot(), getTextArea.get().getCaret().getMark());
			} else {
				len = doc.getLength();
			}
			try {
				String toChange = doc.getText(offset, len);
				doc.remove(offset, len);
				String newString = operation(e.getActionCommand(),toChange);
				doc.insertString(offset, newString, null);
			} catch (BadLocationException ignore) {}
		}
	};
	private JTabbedPane tabs;
	private Supplier<JTextArea> getTextArea = () -> (JTextArea) ((JScrollPane) tabs.getSelectedComponent()).getViewport().getComponents()[0];
	
	public CaseChanger(JTabbedPane tabs) {
		this.tabs = tabs;
		setText("Change case");
		
		JMenuItem item1 = new JMenuItem(action);
		item1.setActionCommand("upper");
		item1.setText("to uppercase");
		add(item1);
		
		JMenuItem item2 = new JMenuItem(action);
		item2.setActionCommand("lower");
		item2.setText("to lowercase");
		add(item2);
		
		JMenuItem item3 = new JMenuItem(action);
		item3.setActionCommand("invert");
		item3.setText("invert case");
		add(item3);
		
		unActivate();
	}
	
	private String operation(String type, String str) {
		if(type.equals("upper")) {
			return str.toUpperCase();
		}
		else if(type.equals("lower")) {
			return str.toLowerCase();
		}
		else {
			StringBuilder b = new StringBuilder();
			for(int i = 0; i < str.length(); ++i) {
				if(Character.isUpperCase(str.charAt(i))) {
					b.append(Character.toLowerCase(str.charAt(i)));
				}
				else {
					b.append(Character.toUpperCase(str.charAt(i)));
				}
			}
			return b.toString();
		}
	}
	
	public void activate() {
		setEnabled(true);
	}
	
	public void unActivate() {
		setEnabled(false);
	}
}


