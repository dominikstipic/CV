package hr.fer.zemris.java.hw11.jnotepadpp;
import static java.lang.Math.abs;

import java.awt.GridLayout;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.function.Supplier;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;
import javax.swing.text.BadLocationException;

public class StatusBar extends JPanel{
	private static final long serialVersionUID = -5672397659771790345L;
	private JTabbedPane tabs;
	private Supplier<JTextArea> getTextArea = () -> (JTextArea) ((JScrollPane) tabs.getSelectedComponent()).getViewport().getComponents()[0];
	private JLabel stats = new JLabel();
	private JLabel clock = new JLabel();
	private DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
	
	public StatusBar(JTabbedPane tabs) {
		this.tabs = tabs;
		setLayout(new GridLayout(1, 2));
		add(stats);
		add(clock);
		startClock();
	}

	public void update(int dot, int mark) {
		int length = getTextArea.get().getText().length();
		int sel = abs(dot - mark);
		try {
			int ln = getTextArea.get().getLineOfOffset(dot);
			int col = dot - getTextArea.get().getLineStartOffset(ln);
			String s = String.format("length:%d        ln:%d Col:%d Sel:%d",length,ln,col,sel);
			stats.setText(s);
		} catch (BadLocationException ignore) {}
		
		
		
	}
	
	public void startClock() {
		Thread t = new Thread(()-> {
			while(true) {
				LocalDateTime now = LocalDateTime.now();
				SwingUtilities.invokeLater(()->{
					clock.setText(dtf.format(now));
				});
				try {
					Thread.sleep(1000);
				} catch (InterruptedException ignore) {}
			}
		});
		t.setDaemon(true);
		t.start();
	}
	
}
