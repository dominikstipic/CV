package hr.fer.nenr.main;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

public class Main extends JFrame{
	private static final long serialVersionUID = 1L;
	JButton east = new JButton("EAST");
	JButton south = new JButton("SOUTH");
	JButton west = new JButton("WEST");
	JButton north = new JButton("NORTH");
	JButton center = new JButton("CENTER");
	private Dimension d;
	
	
	public Main(String appName) {
		initFrame(appName);
		addComponentListener(new ComponentAdapter() {
			@Override
			public void componentResized(ComponentEvent e) {
				d = e.getComponent().getSize();
				updateSize();
			}
		});
		
	}
	
	private void initFrame(String appName) {
		setName(appName);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setSize(700, 700);
		setLocation(new Point(300,100));
		setVisible(true);
		
		
		JPanel panel = new JPanel(new BorderLayout());
		setContentPane(panel);
		
		getContentPane().add(east, BorderLayout.EAST);
		getContentPane().add(west, BorderLayout.WEST);
		getContentPane().add(south, BorderLayout.SOUTH);
		getContentPane().add(north, BorderLayout.NORTH);
		getContentPane().add(center, BorderLayout.CENTER);

	}
	
	public void updateSize() {
		int width = (int) d.getWidth();
		int height = (int) d.getHeight();
		System.out.println(width + " " + height);
		Dimension dim = new Dimension(width/2, 0);
		east.setPreferredSize(dim);
	}
	
	public static void main(String[] args) {
		String appName = "Greeks predictor"; 
		SwingUtilities.invokeLater(() -> new Main(appName));
	}

	
	
}
